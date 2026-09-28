import torch
import torch.nn as nn
import numpy as np
from utils import LoadDataset, UnsupervisedDataset
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from dlmodels import ClassificationNet
from tqdm.notebook import tqdm
from torchvision import transforms
from simclr import SimCLR_Loss, LRScheduler, EarlyStopping
from sklearn.model_selection import train_test_split
import random
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

files_path = [
                './Dataset/federate/client_1_train.h5',
                './Dataset/federate/client_2_train.h5',
                './Dataset/federate/client_3_train.h5',
                './Dataset/federate/client_4_train.h5',
]

LoadDatasetObj = LoadDataset()

tx_range1 = np.arange(0, 5, dtype=int)
tx_range2 = np.arange(0, 10, dtype=int)
tx_range3 = np.arange(0, 15, dtype=int)
tx_range4 = np.arange(0, 20, dtype=int)

tx_ranges = [tx_range1, tx_range2, tx_range3, tx_range4]
quantities = [3, 8, 14, 17]
tx_ranges_rand = []

for i, tx_range in enumerate(tx_ranges):
    # Determine a random quantity of numbers to select
    # quantity = np.random.randint(1, len(tx_range))
    quantity = quantities[i]
    
    # Randomly select numbers from the range
    random_numbers = np.random.choice(tx_range, size=quantity, replace=False)

    tx_ranges_rand.append(random_numbers)
    print(f"For tx_range{i+1}, selected numbers are: {random_numbers}")
print(tx_ranges_rand)

IQ = []
for file, tx_range in zip(files_path, tx_ranges_rand):
    print(f"Loading file: {file}")
    for tx in tx_range:
        start_pkt = 0
        packet_num = random.randint(500, 800)

        end_pkt = start_pkt + packet_num  
        print(f"Device {tx}: end_pkt={end_pkt}, packet_num={packet_num}")
        
        pkt_range = range(start_pkt, end_pkt)
        IQ_indival, _= LoadDatasetObj.load_iq_samples(file,
                                                    [tx],
                                                    pkt_range)
        IQ.append(IQ_indival)

# Convert lists to numpy arrays
IQ = np.concatenate(IQ, axis=0)

data_train, data_valid = train_test_split(IQ,
                                        test_size=0.2,
                                        shuffle=True)
print('Loading done.')

batch_size = 32
dataset_train = UnsupervisedDataset(data_train)
dataset_valid = UnsupervisedDataset(data_valid)

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
bestloss = 1000

criterion_cl = SimCLR_Loss(batch_size, 0.05).to(device)

model = ClassificationNet().to(device) 

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

lr_scheduler = LRScheduler(optimizer)
early_stopping = EarlyStopping()

training_loss = [] 
valid_loss = [] 

start_time = time.time()

for epoch in range(epochs):  # loop over the dataset multiple times

    model.train()
    training_running_loss = []

    # i = 0
    for iteration, (inputs_agu1, inputs_agu2) in enumerate(train_generator):
        # print(f"inputs_agu1.shape={inputs_agu1.shape}, inputs_agu2.shape={inputs_agu2.shape}")
        inputs_agu1 = inputs_agu1.to(device)
        inputs_agu2 = inputs_agu2.to(device)
        optimizer.zero_grad()

        proj_1 = model(inputs_agu1)
        proj_2 = model(inputs_agu2)
        loss_train_cl = criterion_cl(proj_1, proj_2)
        loss_train_cl.backward()
        optimizer.step()

        training_running_loss.append(loss_train_cl.item())
        
    training_epoch_loss = np.mean(training_running_loss)

    model.eval()
    valid_running_loss = []

    for iteration, (inputs_agu1, inputs_agu2) in enumerate(valid_generator):

        inputs_agu1 = inputs_agu1.to(device)
        inputs_agu2 = inputs_agu2.to(device)
        
        proj_1 = model(inputs_agu1)
        proj_2 = model(inputs_agu2)

        loss_val_cl = criterion_cl(proj_1, proj_2)

        # print statistics
        valid_running_loss.append(loss_val_cl.item())
        
    valid_epoch_loss = np.mean(valid_running_loss)


    print('Epoch '+str(epoch+1) 
            +'\t Training Loss: '+str(round(training_epoch_loss, 4))
            +'\t Validation Loss: '+str(round(valid_epoch_loss, 4))
            +'\t')

    if valid_epoch_loss < bestloss:
        bestloss = valid_epoch_loss
        print("model updated!")
        torch.save(model.state_dict(), './saved_models/unsupervise_pretrain.pth')
    
    lr_scheduler(valid_epoch_loss)
    early_stopping(valid_epoch_loss)
    if early_stopping.early_stop:
        break

end_time = time.time()
total_training_time = end_time - start_time

hours = int(total_training_time // 3600)
minutes = int((total_training_time % 3600) // 60)
seconds = total_training_time % 60

print(f"\nFinish！")
print(f"Total training time: {hours}Hours {minutes}Minutes {seconds:.2f}Seconds")
print(f"Total training time: {total_training_time:.2f}Seconds")
