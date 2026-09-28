import copy
import torch
import torch.nn as nn
from utils import to_onehot, Contras_Dataset
import numpy as np
from sklearn.model_selection import train_test_split
import h5py
from torch.utils.data import DataLoader
from simclr import SimCLR_Loss, LRScheduler, EarlyStopping
from dlmodels import ClassificationNet, Clf_Model, ClassificationNet_iq
from tqdm.notebook import tqdm
import random
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_rffi_dataset(file_path):
    
    print(f"Loading RFFI dataset from: {file_path}")
    
    with h5py.File(file_path, 'r') as h5f:
        data = h5f['data'][:]  # shape: (10000, 8192) complex64
        label = h5f['label'][:]  # shape: (10000,) int32
        
        print(f"Dataset shape: {data.shape}")
        print(f"Label shape: {label.shape}")
        print(f"Label range: {label.min()} - {label.max()}")
        print(f"Label distribution: {np.bincount(label)[1:]}")
        
        return data, label



file_path = './Dataset/orgen_dataset/merged_10devices_400pkt.h5'

data, Dev_label = load_rffi_dataset(file_path)

num_per_device = 200
selected_indices = []
for dev in np.unique(Dev_label):
    idx = np.where(Dev_label == dev)[0][:num_per_device]
    selected_indices.extend(idx)
selected_indices = np.array(selected_indices)
data = data[selected_indices]
Dev_label = Dev_label[selected_indices]

Dev_label = Dev_label - 1

label_one_hot, num_class = to_onehot(Dev_label)

print(f'Training data is collected from {num_class} devices.')

data_train, data_valid, label_train, label_valid = train_test_split(
    data, label_one_hot, test_size=0.2, shuffle=True, random_state=42
)

print('Loading done.')
print(f'Training samples: {len(data_train)}')
print(f'Validation samples: {len(data_valid)}')

batch_size = 32

dataset_train = Contras_Dataset(data_train, label_train)
dataset_valid = Contras_Dataset(data_valid, label_valid)

train_generator = DataLoader(
    dataset_train, 
    batch_size=batch_size, 
    shuffle=True,
    drop_last=True, 
    num_workers=0
)

valid_generator = DataLoader(
    dataset_valid, 
    batch_size=batch_size, 
    shuffle=True, 
    drop_last=True,
    num_workers=0
)

epochs = 1000
min_valid_loss = np.inf

criterion_ce = nn.CrossEntropyLoss()
criterion_cl = SimCLR_Loss(batch_size, 0.05).to(device)

premodel = ClassificationNet().to(device) 
model = Clf_Model(premodel, num_class).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

lr_scheduler = LRScheduler(optimizer)
early_stopping = EarlyStopping()

training_loss, training_acc = [], []
valid_loss, valid_acc = [], []

simclr_coef_1 = 1e0
simclr_coef_2 = 1e0

bestloss = 1000

print(f"Starting training for {epochs} epochs...")

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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        optimizer.step()

        training_running_loss.append(loss_train.item())
        acc_tmp = ((preds == truth).sum().item()) / len(truth)
        training_running_correct.append(acc_tmp)
    
    training_epoch_loss = np.mean(training_running_loss)
    training_running_correct = np.mean(training_running_correct)
    
    model.eval()
    valid_running_loss = []
    valid_running_correct = []
    
    with torch.no_grad():
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
        print("Model updated!")
        torch.save(model.state_dict(), f'./saved_models/rffi_oregonstate.pth')
    
    lr_scheduler(valid_epoch_loss)
    early_stopping(valid_epoch_loss)
        

    
