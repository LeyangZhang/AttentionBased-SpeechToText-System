import torch
import torch.nn as nn
import torch.nn.utils as utils
import numpy as np

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class Attention(nn.Module):
    '''
    Attention is calculated using key, value and query from Encoder and decoder.
    Below are the set of operations you need to perform for computing attention:
        energy = bmm(key, query)
        attention = softmax(energy)
        context = bmm(attention, value)
    '''
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, query, key, value, lens_for_attention):
        '''
        :param query :(N, context_size) Query is the output of LSTMCell from Decoder
        :param key: (N, key_size) Key Projection from Encoder per time step
        :param value: (N, value_size) Value Projection from Encoder per time step
        :return output: Attended Context
        :return attention_mask: Attention mask that can be plotted  
        '''

        key = torch.transpose(key, 0, 1)
        value = torch.transpose(value, 0, 1)

        mask = torch.arange(key.size(1)).unsqueeze(0) >= lens_for_attention.unsqueeze(1)
        mask = mask.to(DEVICE)

        energy = torch.bmm(key, query.unsqueeze(2)).squeeze(2)
        energy.masked_fill_(mask, -1e9)
        attention_mask = nn.functional.softmax(energy, dim=1)
        out = torch.bmm(attention_mask.unsqueeze(1), value).squeeze(1)

        return out, attention_mask

class pBLSTM(nn.Module):
    '''
    Pyramidal BiLSTM
    The length of utterance (speech input) can be hundereds to thousands of frames long.
    The Paper reports that a direct LSTM implementation as Encoder resulted in slow convergence,
    and inferior results even after extensive training.
    The major reason is inability of AttendAndSpell operation to extract relevant information
    from a large number of input steps.
    '''
    def __init__(self, input_dim, hidden_dim):
        super(pBLSTM, self).__init__()
        self.blstm1 = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, bidirectional=True)
        self.blstm2 = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, bidirectional=True)
        self.blstm3 = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, bidirectional=True)

    def forward(self, x, lens):
        '''
        :param x :(N, T, H) input to the pBLSTM
        :return output: (N, T, H) encoded sequence from pyramidal Bi-LSTM 
        '''
        # 1
        x = torch.transpose(x, 0, 1)
        x = x[:, :x.shape[1] // 2 * 2, :]

        batch_size = x.shape[0]
        seq_lens = x.shape[1]
        hidden_size = x.shape[2]

        x = x.reshape(batch_size, seq_lens // 2, hidden_size * 2)
        lens //= 2
        x = torch.transpose(x, 0, 1)

        packed_x = utils.rnn.pack_padded_sequence(x, lens, enforce_sorted=False)
        packed_out1 = self.blstm1(packed_x)[0]

        out1, out1_lens = utils.rnn.pad_packed_sequence(packed_out1)

        # 2
        out1 = torch.transpose(out1, 0, 1)
        out1 = out1[:, :out1.shape[1] // 2 * 2, :]

        batch_size = out1.shape[0]
        seq_lens = out1.shape[1]
        hidden_size = out1.shape[2]

        out1 = out1.reshape(batch_size, seq_lens // 2, hidden_size * 2)
        out1_lens //= 2
        out1 = torch.transpose(out1, 0, 1)

        packed_out1 = utils.rnn.pack_padded_sequence(out1, out1_lens, enforce_sorted=False)
        packed_out2 = self.blstm2(packed_out1)[0]

        out2, out2_lens = utils.rnn.pad_packed_sequence(packed_out2)

        # 3
        out2 = torch.transpose(out2, 0, 1)
        out2 = out2[:, :out2.shape[1] // 2 * 2, :]

        batch_size = out2.shape[0]
        seq_lens = out2.shape[1]
        hidden_size = out2.shape[2]

        out2 = out2.reshape(batch_size, seq_lens // 2, hidden_size * 2)
        out2_lens //= 2
        out2 = torch.transpose(out2, 0, 1)

        packed_out2 = utils.rnn.pack_padded_sequence(out2, out2_lens, enforce_sorted=False)
        packed_out3 = self.blstm3(packed_out2)[0]

        out3, out3_lens = utils.rnn.pad_packed_sequence(packed_out3)

        output = utils.rnn.pack_padded_sequence(out3, out3_lens, enforce_sorted=False)

        return output


class Encoder(nn.Module):
    '''
    Encoder takes the utterances as inputs and returns the key and value.
    Key and value are nothing but simple projections of the output from pBLSTM network.
    '''
    def __init__(self, input_dim, hidden_dim, value_size,key_size):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, bidirectional=True)
        
        ### Add code to define the blocks of pBLSTMs! ###
        self.pBLSTM = pBLSTM(input_dim=hidden_dim * 4, hidden_dim=hidden_dim)

        self.key_network = nn.Linear(hidden_dim*2, key_size)
        self.value_network = nn.Linear(hidden_dim*2, value_size)

    def forward(self, x, lens):
        rnn_inp = utils.rnn.pack_padded_sequence(x, lengths=lens, batch_first=False, enforce_sorted=False)
        outputs, _ = self.lstm(rnn_inp)
        out, out_lens = utils.rnn.pad_packed_sequence(outputs) ## max_seq_lens, batch_size, hidden_size; batch_size

        outputs = self.pBLSTM(out, out_lens)
        ### Use the outputs and pass it through the pBLSTM blocks! ###

        linear_input, lens_for_attention = utils.rnn.pad_packed_sequence(outputs)
        keys = self.key_network(linear_input)
        value = self.value_network(linear_input)

        return keys, value, lens_for_attention


class Decoder(nn.Module):
    '''
    As mentioned in a previous recitation, each forward call of decoder deals with just one time step, 
    thus we use LSTMCell instead of LSLTM here.
    The output from the second LSTMCell can be used as query here for attention module.
    In place of value that we get from the attention, this can be replace by context we get from the attention.
    Methods like Gumble noise and teacher forcing can also be incorporated for improving the performance.
    '''
    def __init__(self, vocab_size, hidden_dim, value_size, key_size, isAttended):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.lstm1 = nn.LSTMCell(input_size=hidden_dim + value_size, hidden_size=hidden_dim)
        self.lstm2 = nn.LSTMCell(input_size=hidden_dim, hidden_size=key_size)

        self.isAttended = isAttended
        if (isAttended == True):
            self.attention = Attention()

        self.character_prob = nn.Linear(key_size + value_size, vocab_size)

    def forward(self, key, values, lens_for_attention, text=None, isTrain=True):
        '''
        :param key :(T, N, key_size) Output of the Encoder Key projection layer
        :param values: (T, N, value_size) Output of the Encoder Value projection layer
        :param text: (N, text_len) Batch input of text with text_length
        :param isTrain: Train or eval mode
        :return predictions: Returns the character perdiction probability 
        '''
        batch_size = key.shape[1]

        if (isTrain == True):
            max_len =  text.shape[1]
            embeddings = self.embedding(text)
        else:
            max_len = 250

        predictions = []
        hidden_states = [None, None]
        prediction = torch.ones(batch_size,1, dtype=torch.int32).to(DEVICE) * 33

        for i in range(max_len):
            # * Implement Gumble noise and teacher forcing techniques 
            # * When attention is True, replace values[i,:,:] with the context you get from attention.
            # * If you haven't implemented attention yet, then you may want to check the index and break 
            #   out of the loop so you do you do not get index out of range errors. 


            attention_score = values.mean(dim=0)  # batch_size, value_size

            if (isTrain):
                isTeacherForce = True if np.random.uniform(0, 1, 1) < 0.7 else False
                if isTeacherForce:
                    char_embed = self.embedding(prediction.argmax(dim=-1))
                else:
                    char_embed = embeddings[:, i, :]
            else:
                char_embed = self.embedding(prediction.argmax(dim=-1)) # batch_size, hidden_dim

            inp = torch.cat([char_embed, attention_score], dim=1)
            hidden_states[0] = self.lstm1(inp, hidden_states[0])

            inp_2 = hidden_states[0][0]
            hidden_states[1] = self.lstm2(inp_2, hidden_states[1])

            ### Compute attention from the output of the second LSTM Cell , which is query ###
            output = hidden_states[1][0]

            if self.isAttended:
                attention_score, attention_mask = self.attention(output, key, values, lens_for_attention)

            prediction = self.character_prob(torch.cat([output, attention_score], dim=1))
            predictions.append(prediction.unsqueeze(1))

        return torch.cat(predictions, dim=1)


class Seq2Seq(nn.Module):
    '''
    We train an end-to-end sequence to sequence model comprising of Encoder and Decoder.
    This is simply a wrapper "model" for your encoder and decoder.
    '''
    def __init__(self, input_dim, vocab_size, hidden_dim, value_size=128, key_size=128, isAttended=False):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, value_size=128, key_size=256)
        self.decoder = Decoder(vocab_size, hidden_dim, value_size, key_size, isAttended)

    def forward(self, speech_input, speech_len, text_input=None, isTrain=True):
        key, value, lens_for_attention = self.encoder(speech_input, speech_len)
        if (isTrain == True):
            predictions = self.decoder(key, value, lens_for_attention,text_input)
        else:
            predictions = self.decoder(key, value, lens_for_attention,text=None, isTrain=False)
        return predictions
