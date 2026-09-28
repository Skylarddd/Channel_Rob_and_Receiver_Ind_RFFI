import copy
import torch
import torch.nn as nn
from utils import LoadDataset, to_onehot, Contras_Dataset_rec, Contras_Dataset
import numpy as np
from sklearn.model_selection import train_test_split
import argparse
import time
from torch.utils.data import DataLoader
from simclr import SimCLR_Loss, LRScheduler, EarlyStopping
from dlmodels import Clf_Model, ClassificationNet, ClassificationNet_iq
from tqdm.notebook import tqdm
import random


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


file_path = [
            './Dataset/mul_rec_gx/b210_1_mj.h5',
            './Dataset/mul_rec_gx/b200_1_mj.h5'
            ]

pkt_range = range(0, 800)

LoadDatasetObj = LoadDataset()
tx_range = np.arange(30, 40, dtype=int)
data, Dev_label, Rec_label = LoadDatasetObj.load_multiple_rx_data(file_path,
                                                        tx_range,
                                                        pkt_range)

Dev_label = Dev_label - tx_range[0]

label_one_hot, num_class = to_onehot(Dev_label)
# receiver_labels, _ = to_onehot(Rec_label)

print('Training data is collected from ' + str(num_class) + ' devices.')
data_train, data_valid, label_train, label_valid, receiver_train, receiver_valid = train_test_split(
    data,
    label_one_hot,
    Rec_label, 
    test_size=0.2,
    shuffle=True
)
print('Loading done.')

batch_size = 32
dataset_train = Contras_Dataset_rec(data_train, label_train, receiver_train)  
dataset_valid = Contras_Dataset_rec(data_valid, label_valid, receiver_valid) 

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
criterion_cl = SimCLR_Loss(batch_size, 0.05).to(device)

premodel = ClassificationNet().to(device) 
# premodel.load_state_dict(torch.load('./saved_models/unsupervise_pretrain.pth'))
# premodel.eval()

model = Clf_Model(premodel, num_class).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

lr_scheduler = LRScheduler(optimizer)
early_stopping = EarlyStopping()

training_loss, training_acc = [], []
valid_loss, valid_acc = [], []

simclr_coef_1 = 1e0
simclr_coef_2 = 1e0

bestloss = 1000

for epoch in range(epochs): 
    model.train()
    training_running_loss = []
    training_running_correct = []

    for iteration, (inputs_agu1, inputs_agu2, labels) in enumerate(train_generator):

        inputs_agu1 = inputs_agu1.to(device)
        inputs_agu2 = inputs_agu2.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        pre_label, proj_1 = model(inputs_agu1)
        _, proj_2 = model(inputs_agu2)

        loss_train_cl = criterion_cl(proj_1, proj_2)
        preds = torch.argmax(pre_label, dim=1)
        truth = torch.argmax(labels, dim=1)
        loss_train_ce = criterion_ce(pre_label, labels)

        loss_train = simclr_coef_1 * loss_train_ce + simclr_coef_2 * loss_train_cl
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

    for iteration, (inputs_agu1, inputs_agu2, labels) in enumerate(valid_generator):

        inputs_agu1 = inputs_agu1.to(device)
        inputs_agu2 = inputs_agu2.to(device)
        labels = labels.to(device)

        pre_label, proj_1 = model(inputs_agu1)
        _, proj_2 = model(inputs_agu2)

        loss_val_cl = criterion_cl(proj_1, proj_2)

        preds = torch.argmax(pre_label, dim=1)
        truth = torch.argmax(labels, dim=1)

        loss_val_ce = criterion_ce(pre_label, labels)
        loss_val = simclr_coef_1 * loss_val_ce + simclr_coef_2 * loss_val_cl


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
            torch.save(model.state_dict(), f'./saved_models/b200b210_speccl.pth')
    
    
    lr_scheduler(valid_epoch_loss)
    early_stopping(valid_epoch_loss)

