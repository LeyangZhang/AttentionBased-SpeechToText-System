import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import *

'''
Loading all the numpy files containing the utterance information and text information
'''
def load_data():
    speech_train = np.load('train_new.npy', allow_pickle=True, encoding='bytes')
    speech_valid = np.load('dev_new.npy', allow_pickle=True, encoding='bytes')
    speech_test = np.load('test_new.npy', allow_pickle=True, encoding='bytes')

    transcript_train = np.load('./train_transcripts.npy', allow_pickle=True,encoding='bytes')
    transcript_valid = np.load('./dev_transcripts.npy', allow_pickle=True,encoding='bytes')

    return speech_train, speech_valid, speech_test, transcript_train, transcript_valid


'''
Transforms alphabetical input to numerical input, replace each letter by its corresponding 
index from letter_list
'''
def transform_letter_to_index(transcript, letter_list):
    '''
    :param transcript :(N, ) Transcripts are the text input
    :param letter_list: Letter list defined above
    :return letter_to_index_list: Returns a list for all the transcript sentence to index
    '''

    letter_dict = dict(zip(letter_list, np.arange(len(letter_list))))
    index_list = []
    for i in range(transcript.shape[0]):
        index_sub_list = []
        index_sub_list.append(letter_dict['<sos>'])
        # x is word
        for x in transcript[i]:
            # xx is character
            for xx in x.decode("utf-8"):
                index_sub_list.append(letter_dict[xx])
            index_sub_list.append(letter_dict[' '])
        index_sub_list.pop()
        index_sub_list.append(letter_dict['<eos>'])
        index_list.append(np.array(index_sub_list))

    return np.array(index_list)


'''
Optional, create dictionaries for letter2index and index2letter transformations
'''
def create_dictionaries(letter_list):
    letter2index = dict()
    index2letter = dict()
    return letter2index, index2letter


class Speech2TextDataset(Dataset):
    '''
    Dataset class for the speech to text data, this may need some tweaking in the
    getitem method as your implementation in the collate function may be different from
    ours. 
    '''
    def __init__(self, speech, text=None, isTrain=True):
        self.speech = [seq for seq in speech]
        self.speech_lens = torch.tensor([len(seq) for seq in speech])
        self.isTrain = isTrain
        if text is not None:
            self.text = [seq for seq in text]
            self.text_lens = torch.tensor([len(seq) for seq in text])

    def __len__(self):
        return len(self.speech)

    def __getitem__(self, index):
        if self.isTrain:
            return torch.tensor(self.speech[index].astype(np.float32)), torch.tensor(self.text[index]), \
                    self.speech_lens[index], self.text_lens[index]

        else:
            return torch.tensor(self.speech[index].astype(np.float32)), torch.tensor(self.speech_lens[index])


def collate_train(batch_data):
    ### Return the padded speech and text data, and the length of utterance and transcript ###
    inputs, targets, len_x, len_y = zip(*batch_data)
    inputs = pad_sequence(inputs)
    targets = pad_sequence(targets, batch_first=True)

    return inputs, targets, torch.LongTensor(len_x), torch.LongTensor(len_y)


def collate_test(batch_data):
    ### Return padded speech and length of utterance ###
    inputs, len_x = zip(*batch_data)
    inputs = pad_sequence(inputs)

    return inputs, torch.LongTensor(len_x)