import copy
import torch
import torch.nn as nn
from utils import LoadDataset, to_onehot, Contras_Dataset_rec, SupervisedDataset_rec
import numpy as np
from sklearn.model_selection import train_test_split
import argparse
import time
from torch.utils.data import DataLoader
from simclr import LRScheduler, EarlyStopping
from dlmodels import ClassificationNet, Clf_Model, Clf_Model_reverse
from tqdm.notebook import tqdm
import random


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

file_path = [
            './Dataset/mul_rec_gx/n210_1_mj.h5',
            './Dataset/mul_rec_gx/b200_mini_1_mj.h5'
            ]

pkt_range = range(0, 800)

LoadDatasetObj = LoadDataset()
# tx_range = np.arange(35, 45, dtype=int)
tx_range = np.arange(30, 40, dtype=int)
data, Dev_label, Rec_label = LoadDatasetObj.load_multiple_rx_data(file_path,
                                                        tx_range,
                                                        pkt_range)

Dev_label = Dev_label - tx_range[0]
# print(Rec_label)

label_one_hot, num_class = to_onehot(Dev_label)
receiver_labels, num_rx = to_onehot(Rec_label)

print('Training data is collected from ' + str(num_class) + ' devices.')
# data_train, data_valid, label_train, label_valid = train_test_split(data,
#                                                                     label_one_hot,
#                                                                     test_size=0.2,
#                                                                     shuffle=True)
data_train, data_valid, label_train, label_valid, receiver_train, receiver_valid = train_test_split(
    data,
    label_one_hot,
    receiver_labels, 
    test_size=0.2,
    shuffle=True
)
print('Loading done.')

batch_size = 32
dataset_train = SupervisedDataset_rec(data_train, label_train, receiver_train)  
dataset_valid = SupervisedDataset_rec(data_valid, label_valid, receiver_valid) 


train_generator = DataLoader(dataset_train, 
                            batch_size=batch_size, 
                            shuffle=True,
                            drop_last=True, 
                            num_workers=0)

valid_generator = DataLoader(dataset_valid, 
                            batch_size=batch_size, 
                            shuffle=True, 
                            drop_last=True,
                            num_workers=0)

epochs = 1000
min_valid_loss = np.inf

criterion_ce = nn.CrossEntropyLoss()


premodel = ClassificationNet().to(device) 
model = Clf_Model_reverse(premodel, num_class, num_rx).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

lr_scheduler = LRScheduler(optimizer)
early_stopping = EarlyStopping()

training_loss, training_acc = [], []
valid_loss, valid_acc = [], []

bestloss = 1000

for epoch in range(epochs):  
    model.train()
    training_running_loss = []
    training_running_correct = []

    for iteration, (inputs_agu1, labels, rxlabels) in enumerate(train_generator):
        # print shape
        # print('inputs_agu1', inputs_agu1.shape)
        inputs_agu1 = inputs_agu1.to(device)
        labels = labels.to(device)
        rxlabels = rxlabels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        pre_label, rx = model(inputs_agu1)

        labels = labels.float()
        rxlabels = rxlabels.float()
        
        preds = torch.argmax(pre_label, dim=1)
        truth = torch.argmax(labels, dim=1)
        pre_rx = torch.argmax(rx, dim=1)
        truth_rx = torch.argmax(rxlabels, dim=1)
        

        
        loss_train_tx = criterion_ce(pre_label, labels)
        loss_train_rx = criterion_ce(rx, rxlabels)
        
        loss_train = loss_train_tx + loss_train_rx

        loss_train.backward()
        optimizer.step()

        training_running_loss.append(loss_train.item())
        acc_tmp = ((preds == truth).sum().item()) / len(truth)
        training_running_correct.append(acc_tmp)


    training_epoch_loss = np.mean(training_running_loss)
    training_running_correct = np.mean(training_running_correct)

    model.eval()
    valid_running_loss = []
    valid_running_correct = []

    for iteration, (inputs_agu1, labels, rxlabels) in enumerate(valid_generator):

        inputs_agu1 = inputs_agu1.to(device)
        labels = labels.to(device)
        rxlabels = rxlabels.to(device)
        labels = labels.float()
        rxlabels = rxlabels.float()

        pre_label, rx = model(inputs_agu1)

        preds = torch.argmax(pre_label, dim=1)
        truth = torch.argmax(labels, dim=1)

        pre_rx = torch.argmax(rx, dim=1)
        truth_rx = torch.argmax(rxlabels, dim=1)
        loss_rx = criterion_ce(rx, rxlabels)

        loss_val_tx = criterion_ce(pre_label, labels)
        loss_val_rx = criterion_ce(rx, rxlabels)
        
        loss_val = loss_val_tx + loss_val_rx

        # print statistics
        valid_running_loss.append(loss_val.item())
        acc_tmp = ((preds == truth).sum().item()) / len(truth)
        valid_running_correct.append(acc_tmp)


    valid_epoch_loss = np.mean(valid_running_loss)
    valid_running_correct = np.mean(valid_running_correct)


    print('Epoch '+str(epoch+1)
            +'\t Training Loss: '+str(round(training_epoch_loss, 4))
            +'\t Validation Loss: '+str(round(valid_epoch_loss, 4))
            +'\t Training Acc: '+str(round(training_running_correct,4))
            +'\t Validation Acc: '+str(round(valid_running_correct, 4))
            +'\t')

    if valid_epoch_loss < bestloss:
            bestloss = valid_epoch_loss
            print("model updated!") 
            torch.save(model.state_dict(), f'./saved_models/n2101b200mini_spec_reverse_{s+1}.pth')

    
    lr_scheduler(valid_epoch_loss)
    early_stopping(valid_epoch_loss)

