import torch
import torch.nn as nn
import numpy as np
from utils4090 import LoadDataset, signal_preprocess_for_cnn_2, normalization, fourmension
import sr_pytorch as sr
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from dlmodels4090 import  ClassificationNet_iq, Clf_Model, ClassificationNet
from tqdm.notebook import tqdm
from scipy import signal
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

fig = plt.figure(layout='constrained', figsize=(20, 20))
subfigs = fig.subfigures(3, 3, wspace=0.00)

win_len = 256
crop_ratio = 0.3

pkt_range=range(0, 200)
tx_range = np.arange(30, 40, dtype=int)

data_dir = Path("./Dataset/mul_rec_gx")

data_path_lopy_set = sorted(data_dir.glob("*_test.h5"))

premodel = ClassificationNet().to(device) 
# premodel = ClassificationNet_iq().to(device) 
# premodel.eval()
# model = Clf_Model_reverse(premodel, 10, 2).to(device)
model = Clf_Model(premodel, 10).to(device)

model.load_state_dict(torch.load('./saved_models/b200mini1n2101_speccl.pth'))
model.eval()

acc_set = []
for i, data_path_lopy in enumerate(data_path_lopy_set):
    index_i, index_j = i % 3, i // 3
    
    LoadDatasetObj = LoadDataset()

    [IQ_data_lopy, Dev_label_lopy] = LoadDatasetObj.load_iq_samples(data_path_lopy, 
                                                            tx_range, pkt_range)

    IQ_data = np.concatenate([IQ_data_lopy])
    Dev_label_lopy = Dev_label_lopy - tx_range[0]
    Dev_label = np.concatenate([Dev_label_lopy])

    classes_num = len(np.unique(Dev_label))
    data_test = normalization(IQ_data)
    
    # spectrogram
    spec = sr.spectrogram(data_test, win_len, crop_ratio)
    spec = signal_preprocess_for_cnn_2(spec).to(device)
    
    # IQ data
    # spec = fourmension(data_test)
    # spec = torch.from_numpy(spec)
    # spec = spec.to(device).float()

    with torch.no_grad():
        clf_output, _  = model(spec)
        # clf_output, _, _  = model(spec)

    label = torch.from_numpy(Dev_label)
    label = label.to(device).squeeze()

    acc = accuracy_score(label.cpu(), clf_output.cpu().argmax(dim=1))
    # print(acc)
    acc_set.append(acc)
    
    del IQ_data_lopy, Dev_label_lopy, IQ_data, Dev_label, data_test, spec, clf_output, label
    torch.cuda.empty_cache()

print(acc_set)