import time
import torch
import torch.nn as nn
import torch.distributions
from models import *

### Add Your Other Necessary Imports Here! ###

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

MAX_LEN = 250
END_INDEX = 34

LETTER_LIST = ['<pad>', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', \
               'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '-', "'", '.', '_', '+', ' ','<sos>','<eos>']

def train(model, train_loader, criterion, optimizer, epoch):
    model.train()
    model.to(DEVICE)
    start = time.time()
    perplexity_loss = 0

    # 1) Iterate through your loader
    for speech, text, speech_lens, text_lens in train_loader:

        # 2) Use torch.autograd.set_detect_anomaly(True) to get notices about gradient explosion
        # torch.autograd.set_detect_anomaly(True)

        # 3) Set the inputs to the device.
        model = model.to(DEVICE)
        criterion = criterion.to(DEVICE)
        speech = speech.to(DEVICE)  # max_seq_len, batch_size, 40
        text = text.to(DEVICE)   #  max_seq_len, batch_size
        # speech_lens = speech_lens.to(DEVICE)
        # text_lens = text_lens.to(DEVICE)

        # 4) Pass your inputs, and length of speech into the model.
        output = model(speech, speech_lens, text)

        # 5) Generate a mask based on the lengths of the text to create a masked loss.
        # 5.1) Ensure the mask is on the device and is the correct shape.
        output_mask = torch.arange(text.shape[1]).unsqueeze(0) < text_lens.unsqueeze(1)
        output_mask = output_mask.to(DEVICE)
        optimizer.zero_grad()

        loss = 0
        n_tokens = text_lens.sum()

        for i in range(text.size(1) - 1):
            cur_output = output[:, i, :]
            active = output_mask[:, i]

            if loss == 0:
                loss = criterion(cur_output[active], text[active, i+1])
            else:
                loss += criterion(cur_output[active], text[active, i+1])

        loss /= n_tokens
        loss.backward()

        perplexity_loss = torch.exp(loss)

        optimizer.step()

            # 6) If necessary, reshape your predictions and origianl text input 
            # 6.1) Use .contiguous() if you need to. 

            # 7) Use the criterion to get the loss.

            # 8) Use the mask to calculate a masked loss. 

            # 9) Run the backward pass on the masked loss. 

            # 10) Use torch.nn.utils.clip_grad_norm(model.parameters(), 2)
            
            # 11) Take a step with your optimizer

            # 12) Normalize the masked loss

            # 13) Optionally print the training loss after every N batches

    end = time.time()
    print('time comsuming is ', end - start)
    print('Training perplexity loss is ', perplexity_loss)

    return perplexity_loss

def val(model, valid_loader, criterion, optimizer, epoch):
    model.to(DEVICE)
    start = time.time()
    perplexity_loss = 0

    for speech, text, speech_lens, text_lens in valid_loader:
        model = model.to(DEVICE)
        criterion = criterion.to(DEVICE)
        speech = speech.to(DEVICE)  # max_seq_len, batch_size, 40
        text = text.to(DEVICE)  # max_seq_len, batch_size

        output = model(speech, speech_lens, text)
        output_mask = torch.arange(text.shape[1]).unsqueeze(0) < text_lens.unsqueeze(1)
        output_mask = output_mask.to(DEVICE)
        optimizer.zero_grad()

        loss = 0
        n_tokens = text_lens.sum()

        for i in range(text.size(1) - 1):
            cur_output = output[:, i, :]
            active = output_mask[:, i]

            if loss == 0:
                loss = criterion(cur_output[active], text[active, i + 1])
            else:
                loss += criterion(cur_output[active], text[active, i + 1])

        loss /= n_tokens
        perplexity_loss = torch.exp(loss)

    end = time.time()
    print('time comsuming is ', end - start)
    print('valid perplexity loss is ', perplexity_loss)

    return perplexity_loss


def test(model, test_loader):
    ### Write your test code here! ###

    model.to(DEVICE)
    greedy_path_final = []

    for speech, speech_lens in test_loader:
        # speech: seq_len, batch_size, 40
        # speech_lens: batch_size

        batch_size = speech.size(1)

        speech = speech.to(DEVICE)

        output = model(speech, speech_lens, text_input=None, isTrain=False)
        output_prob = nn.functional.softmax(output, dim=2)

        greedy_path = torch.argmax(output_prob, dim=2)


        for i in range(batch_size):

            end_index = (greedy_path[i, :] == END_INDEX).nonzero().squeeze(-1)
            if end_index.shape[0] == 0:
                greedy_path_final.append(greedy_path[i, :])
            else:
                greedy_path_final.append(greedy_path[i, :end_index[0]])

    return greedy_path_final


def load_model(model, test_loader):

    model.load_state_dict(torch.load('model3'))
    model.eval()
    model = model.to(DEVICE)

    predicted_text_index = test(model, test_loader)

    seq_string_list = []

    for i in range(len(predicted_text_index)):

        seq_string = "".join([LETTER_LIST[y] for y in predicted_text_index[i]])
        seq_string_list.append(seq_string)

    import pandas as pd
    my_df = pd.DataFrame(seq_string_list)
    my_df.to_csv('pred.csv', index=True, header=False)

    return seq_string_list