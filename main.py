import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models import Seq2Seq
from train_test import train, test
from dataloader import load_data, collate_train, collate_test, transform_letter_to_index, Speech2TextDataset

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

LETTER_LIST = ['<pad>', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', \
               'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '-', "'", '.', '_', '+', ' ','<sos>','<eos>']

def main():
    model = Seq2Seq(input_dim=40, vocab_size=len(LETTER_LIST), hidden_dim=128)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(reduction=None)
    nepochs = 25
    batch_size = 64 if DEVICE == 'cuda' else 1

    speech_train, speech_valid, speech_test, transcript_train, transcript_valid = load_data()
    character_text_train = transform_letter_to_index(transcript_train, LETTER_LIST)
    character_text_valid = transform_letter_to_index(transcript_valid, LETTER_LIST)

    train_dataset = Speech2TextDataset(speech_train, character_text_train)
    # val_dataset = 
    test_dataset = Speech2TextDataset(speech_test, None, False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_train)
    # val_loader = 
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_test)

    for epoch in range(nepochs):
        train(model, train_loader, criterion, optimizer, epoch)
        # val()
        test(model, test_loader, epoch)


if __name__ == '__main__':
    ## main()
    ## valid data
    ## x, array of array, (frames 1106, time_step, 40)
    ## y, array of array, (frames 1106, sentence_len)
    speech_train, speech_valid, speech_test, transcript_train, transcript_valid = load_data()
    print(speech_train.shape)