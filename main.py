import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models import Seq2Seq
from train_test import train, test, val, load_model
from dataloader import load_data, collate_train, collate_test, transform_letter_to_index, Speech2TextDataset

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

LETTER_LIST = ['<pad>', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', \
               'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '-', "'", '.', '_', '+', ' ','<sos>','<eos>']

def main():
    model = Seq2Seq(input_dim=40, vocab_size=len(LETTER_LIST), hidden_dim=128, value_size=128, key_size=256, isAttended=True)
    model.load_state_dict(torch.load('model3'))
    model.eval()
    model = model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.CrossEntropyLoss(reduction='sum')
    nepochs = 10
    batch_size = 64 if DEVICE == 'cuda' else 1

    speech_train, speech_valid, speech_test, transcript_train, transcript_valid = load_data()
    character_text_train = transform_letter_to_index(transcript_train, LETTER_LIST)
    character_text_valid = transform_letter_to_index(transcript_valid, LETTER_LIST)

    train_dataset = Speech2TextDataset(speech_train, character_text_train)
    val_dataset = Speech2TextDataset(speech_valid, character_text_valid)
    test_dataset = Speech2TextDataset(speech_test, None, False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_train)
    for x in train_loader:
        pass

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_train)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_test)

    for epoch in range(nepochs):
        print('==============', 'Epoch', epoch+1,'================')
        train(model, train_loader, criterion, optimizer, epoch)
        val(model, val_loader, criterion, optimizer, epoch)
    torch.save(model.state_dict(), 'model3')

    load_model(model, test_loader)

if __name__ == '__main__':
    main()
    print()

    # valid data
    # x, array of array, (frames 1106, time_step, 40)
    # y, array of array, (frames 1106, sentence_len)
    # print(DEVICE)
    # speech_train, speech_valid, speech_test, transcript_train, transcript_valid = load_data()
    # trans_index_valid = transform_letter_to_index(transcript_valid, LETTER_LIST)
    # val_Dataset = Speech2TextDataset(speech_valid, trans_index_valid)
    # print(val_Dataset[5][0].shape)
    # print(val_Dataset[5][1].shape)
    # print(val_Dataset[5][2])
    # print(val_Dataset[5][3])
    # train_loader = DataLoader(val_Dataset, shuffle=False, batch_size=10, collate_fn=collate_train)
    # for inputs, targets, inputs_lens, targets_lens in train_loader:
    #     print(inputs_lens)
    #     print(targets_lens)
    #     break



