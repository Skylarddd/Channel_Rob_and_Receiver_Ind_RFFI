import numpy as np
from torch.nn import functional as F
import h5py
import torch.nn as nn
import random
from numpy import sum, sqrt
from numpy.random import standard_normal, uniform
import math
from scipy import signal
import torch
import sr_pytorch as sr
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patheffects as PathEffects
from torch.utils.data import Dataset
from pyphysim.channels.fading import TdlChannel, TdlChannelProfile
from pyphysim.channels.fading_generators import JakesSampleGenerator, RayleighSampleGenerator
from sklearn.metrics import confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LoadDataset:
    def __init__(
        self,
    ):
        self.dataset_name = "data"
        self.labelset_name = "label"

    def _convert_to_complex(self, data):
        """Convert the loaded data to complex IQ samples."""
        num_row = data.shape[0]
        num_col = data.shape[1]
        data_complex = np.zeros([num_row, round(num_col / 2)], dtype=complex)

        data_complex = (
            data[:, : round(num_col / 2)] + 1j * data[:, round(num_col / 2) :]
        )
        return data_complex

    def load_iq_samples(self, file_path, dev_range, pkt_range):
        """
        Load IQ samples from a dataset.

        INPUT:
            FILE_PATH is the dataset path.

            DEV_RANGE specifies the loaded device range.

            PKT_RANGE specifies the loaded packets range.

        RETURN:
            DATA is the laoded complex IQ samples.

            LABLE is the true label of each received packet.
        """

        f = h5py.File(file_path, "r")
        label = f[self.labelset_name][:]
        label = label.astype(int)
        label = np.transpose(label)
        label = label - 1

        # label_start = int(label[0]) + 1
        # label_end = int(label[-1]) + 1
        # num_dev = label_end - label_start + 1
        # num_pkt = len(label)
        # num_pkt_per_dev = int(num_pkt / num_dev)

        # print(
        #     "Dataset information: Dev "
        #     + str(label_start)
        #     + " to Dev "
        #     + str(label_end)
        #     + ", "
        #     + str(num_pkt_per_dev)
        #     + " packets per device."
        # )

        sample_index_list = []

        for dev_idx in dev_range:
            sample_index_dev = np.where(label == dev_idx)[0][pkt_range].tolist()
            sample_index_list.extend(sample_index_dev)

        data = f[self.dataset_name][sample_index_list]
        data = self._convert_to_complex(data)

        label = label[sample_index_list]

        f.close()
        return data, label

    def load_multiple_rx_data(self, file_list, tx_range, pkt_range):

        num_rx = len(file_list)
        num_tx = len(tx_range)
        num_pkt = len(pkt_range)

        data = []
        tx_label = []
        rx_label = []

        for file_idx in range(num_rx):
            print("Start loading dataset " + str(file_idx + 1))
            filename = file_list[file_idx]
            # filename = folder_name + filename
            [data_temp, tx_label_temp] = self.load_iq_samples(
                filename, tx_range, pkt_range
            )
            rx_label_temp = np.ones(num_pkt * num_tx) * file_idx

            data.extend(data_temp)
            tx_label.extend(tx_label_temp)
            rx_label.extend(rx_label_temp)
            print("Finish loading dataset " + str(file_idx + 1))

        data = np.array(data)
        tx_label = np.array(tx_label)
        rx_label = np.array(rx_label)

        return data, tx_label, rx_label
    
    
class UnsupervisedDataset(Dataset):
    def __init__(self, data_input):
        self.data_input = data_input

    def __len__(self):
        return len(self.data_input)

    def __getitem__(self, index):
        data_input = self.data_input[index]

        data_left = self._process_data(data_input)
        data_right = self._process_data(data_input)

        return data_left, data_right

    def _process_data(self, data_in):
        data_out = data_aug_operator(data_in, multipath=random.randint(0, 1))
        # data_out = data_aug_operator(data_in, multipath=False)
        data_out = normalization(data_out)
        # data_out = channel_ind_spectrogram(data_out[0], win_len=256, crop_ratio=0.3)
        data_out = spectrogram(data_out[0], win_len=256, crop_ratio=0.3)
        data_out = torch.from_numpy(data_out.astype(np.float32))
        # data_out = data_out.permute(0, 3, 1, 2)
        
        # return data_out.squeeze(0)
        return data_out


class SupervisedDataset(Dataset):
    def __init__(self, data_input, label_input):
        self.data_input = data_input
        self.label_input = label_input

    def __len__(self):
        return len(self.data_input)

    def __getitem__(self, index):
        data_input = self.data_input[index]
        label_input = self.label_input[index]

        data_input= self._process_data(data_input)
        label_input = label_input.astype(float)
        return data_input, label_input

    def _process_data(self, data_in):
        data_out = data_aug_operator(data_in, multipath=random.randint(0, 1))
        # data_out = data_aug_operator(data_in, multipath=False)
        data_out = normalization(data_out)
        data_out = spectrogram(data_out[0], win_len=256, crop_ratio=0.3)
        # data_out = channel_ind_spectrogram(data_out[0], win_len=256, crop_ratio=0.3)
        # data_out = fourmension(data_out)
        # data_out = data_out[0]
        # print(data_out.shape)
        data_out = torch.from_numpy(data_out.astype(np.float32))

        return data_out

class SupervisedDataset_rec(Dataset):
    def __init__(self, data_input, label_input, rec_input):
        self.data_input = data_input
        self.label_input = label_input
        self.rec_input = rec_input

    def __len__(self):
        return len(self.data_input)

    def __getitem__(self, index):
        data_input = self.data_input[index]
        label_input = self.label_input[index]
        rec_input = self.rec_input[index]

        data_input= self._process_data(data_input)
        label_input = label_input.astype(float)
        return data_input, label_input, rec_input

    def _process_data(self, data_in):
        data_out = data_aug_operator(data_in, multipath=random.randint(0, 1))
        # data_out = data_aug_operator(data_in, multipath=False)
        data_out = normalization(data_out)
        data_out = spectrogram(data_out[0], win_len=256, crop_ratio=0.3)
        # data_out = channel_ind_spectrogram(data_out[0], win_len=256, crop_ratio=0.3)
        # data_out = fourmension(data_out)
        # data_out = data_out[0]
        # print(data_out.shape)
        data_out = torch.from_numpy(data_out.astype(np.float32))

        return data_out
    
class SupervisedDataset_rec_iq(Dataset):
    def __init__(self, data_input, label_input, rec_input):
        self.data_input = data_input
        self.label_input = label_input
        self.rec_input = rec_input

    def __len__(self):
        return len(self.data_input)

    def __getitem__(self, index):
        data_input = self.data_input[index]
        label_input = self.label_input[index]
        rec_input = self.rec_input[index]

        data_input= self._process_data(data_input)
        label_input = label_input.astype(float)
        return data_input, label_input, rec_input

    def _process_data(self, data_in):
        data_out = data_aug_operator(data_in, multipath=random.randint(0, 1))
        # data_out = data_aug_operator(data_in, multipath=False)
        data_out = normalization(data_out)
        # data_out = spectrogram(data_out[0], win_len=256, crop_ratio=0.3)
        # data_out = channel_ind_spectrogram(data_out[0], win_len=256, crop_ratio=0.3)
        data_out = fourmension(data_out)
        data_out = data_out[0]
        # print(data_out.shape)
        data_out = torch.from_numpy(data_out.astype(np.float32))

        return data_out
    
    
def data_aug_operator(data_in, multipath=True):

    if multipath:
        # data_out = np.zeros(data_in.shape, dtype=complex)
        Ts = 1/1000000
            
        tau_d = np.random.uniform(5, 300)*1e-9
        Fd = np.random.uniform(0, 5)
        # Create a jakes object with 20 rays. This is the fading model that controls how the channel vary in time.
        # This will be passed to the TDL channel object.
        # L:The number of rays for the Jakes model.
        chObj = JakesSampleGenerator(Fd=Fd, Ts=Ts, L=5)
        # chObj = RayleighSampleGenerator()
        avgPathGains, pathDelays = cal_exponential_pdp(tau_d, Ts)

        # Creates the tapped delay line (TDL) channel model, which accounts for the multipath and thus the
        # frequency selectivity
        pdpObj = TdlChannelProfile(avgPathGains,
                                pathDelays,
                                'Exponential_PDP')

        tdlchannel = TdlChannel(chObj, pdpObj)

        data_corrputed = tdlchannel.corrupt_data(data_in)

        data_out = data_corrputed[:len(data_corrputed)-tdlchannel.num_taps+1]

        data_out = awgn(data_out, snr_range = range(10, 40))

    else:
        data_out = awgn(data_in, snr_range = range(10, 40))
        # if len(data_in.shape) == 1:
        #     data_in = data_in.reshape(1, len(data_in))
        # data_out = data_in

    return data_out

def data_aug_operator_test(data_in, snr):

    if snr is not None:
        data_out = awgn(data_in, snr_range = range(snr, snr+1))
        # if len(data_in.shape) == 1:
        #     data_in = data_in.reshape(1, len(data_in))
        # data_out = data_in

    return data_out


def awgn(data, snr_range):
    if len(data.shape) == 1:
        data = data.reshape(1, len(data))

    pkt_num = data.shape[0]
    data_awgn = np.zeros(data.shape, dtype=complex)
    SNRdB = uniform(snr_range[0],snr_range[-1],pkt_num)
    for pktIdx in range(pkt_num):
        s = data[pktIdx]
        SNR_linear = 10**(SNRdB[pktIdx]/10)
        P= sum(abs(s)**2)/len(s)
        N0=P/SNR_linear
        n = sqrt(N0/2)*(standard_normal(len(s))+1j*standard_normal(len(s)))
        data_awgn[pktIdx] = s + n

    # return data, data_awgn
    return data_awgn



def normalization(data):
    ''' Normalize the signal.'''
    
    amplitude = np.abs(data+10e-12)
    rms = np.sqrt(np.mean(amplitude**2))
    data_norm = data/rms
    
    return data_norm


def channel_ind_spectrogram(data, win_len = 256, crop_ratio = 0.3):
    ''' Generate channel independent spectrogram.'''
    def _spec_crop(x, crop_ratio):
    
        num_row = x.shape[0]
        x_cropped = x[math.floor(num_row*crop_ratio):math.ceil(num_row*(1-crop_ratio))]

        return x_cropped

    f, t, spec = signal.stft(data, 
                            window='boxcar', 
                            nperseg= win_len, 
                            noverlap= round(0.5*win_len), 
                            nfft= win_len,
                            return_onesided=False, 
                            padded = False, 
                            boundary = None)
    
    # spec = spec_shift(spec)
    spec = np.fft.fftshift(spec, axes=0)
    # spec = spec_crop(spec, crop_ratio)
    spec = spec + 1e-12
    
    data_out = spec[:,1:]/spec[:,:-1]    
                
    data_out = np.log10(np.abs(data_out)**2)

    data_out = _spec_crop(data_out, crop_ratio)
    
    data_out = np.expand_dims(data_out, axis=0)

    return data_out

def spectrogram(data, win_len = 256, crop_ratio = 0.3):

    def _spec_crop(x, crop_ratio):
        # x = np.squeeze(x, axis=0)
        num_row = x.shape[0]
        x_cropped = x[math.floor(num_row*crop_ratio):math.ceil(num_row*(1-crop_ratio))]
        return x_cropped
    data = normalization(data)
    f, t, spec = signal.stft(data, 
                            window='boxcar', 
                            nperseg= win_len, 
                            noverlap= round(0.5*win_len), 
                            nfft= win_len,
                            return_onesided=False, 
                            padded = False, 
                            boundary = None)

    # spec = spec_shift(spec)
    spec = np.fft.fftshift(spec, axes=0)
    spec = _spec_crop(spec, crop_ratio)
    data_out = np.log10(np.abs(spec+1e-12)**2)
    data_out = np.expand_dims(data_out, axis=0)

    return data_out


def cal_exponential_pdp(tau_d, Ts, A_dB = -30):

    # Exponential PDP generator
    # Inputs:
    # tau_d : rms delay spread[sec]
    # Ts : Sampling time[sec]
    # A_dB : smallest noticeable power[dB]
    # norm_flag : normalize total power to unit
    # Output:
    # PDP : PDP vector

    sigma_tau = tau_d
    A = 10**(A_dB / 10)
    lmax = np.ceil(-tau_d * np.log(A) / Ts)
    
    # Exponential PDP
    p = np.arange(0, lmax+1)
    pathDelays = p * Ts
    
    p = (1 / sigma_tau) * np.exp(-p * Ts / sigma_tau)
    p_norm = p / np.sum(p)
    p_norm = p_norm + 1e-12

    avgPathGains = 10 * np.log10(p_norm)
    
    return avgPathGains, pathDelays

class Contras_Dataset(Dataset):
    def __init__(self, data_input, label_input):
        self.data_input = data_input
        self.label_input = label_input
        
    def __len__(self):
        return len(self.data_input)


    def __getitem__(self, index):
        data_input = self.data_input[index]
        label_input = self.label_input[index]

        data_agu1= self._process_data(data_input)
        data_agu2= self._process_data(data_input)
        label_input = label_input.astype(float)
        
        # view_spec(data_agu1[0])
        # print(data_agu1)
        return data_agu1, data_agu2, label_input

    def _process_data(self, data_in):

        data_out = data_aug_operator(data_in, multipath=random.randint(0, 1))
        # data_out = data_aug_operator(data_in, multipath=False)
        data_out = normalization(data_out)

        # data_out = channel_ind_spectrogram(data_out[0], win_len=256, crop_ratio=0.3)
        # data_out = spectrogram(data_out[0], win_len=256, crop_ratio=0.3)

        
        data_out = fourmension(data_out)
        data_out = data_out[0]
        # # view_spec(data_out[0])
        data_out = torch.from_numpy(data_out.astype(np.float32))

        return data_out
    
class Contras_Dataset_test(Dataset):
    def __init__(self, data_input, label_input, snr):
        self.data_input = data_input
        self.label_input = label_input
        self.snr = snr  
        
    def __len__(self):
        return len(self.data_input)


    def __getitem__(self, index):
        data_input = self.data_input[index]
        label_input = self.label_input[index]
        data_agu1= self._process_data(data_input, self.snr)
        data_agu2= self._process_data(data_input, self.snr)
        label_input = label_input.astype(float)
        
        # view_spec(data_agu1[0])
        # print(data_agu1)
        return data_agu1, data_agu2, label_input

    def _process_data(self, data_in, snr):

        # data_out = data_aug_operator(data_in, multipath=random.randint(0, 1))
        data_out = data_aug_operator_test(data_in, snr)
        data_out = normalization(data_out)

        # data_out = channel_ind_spectrogram(data_out[0], win_len=256, crop_ratio=0.3)
        # data_out = spectrogram(data_out[0], win_len=256, crop_ratio=0.3)

        
        data_out = fourmension(data_out)
        data_out = data_out[0]
        # # view_spec(data_out[0])
        data_out = torch.from_numpy(data_out.astype(np.float32))

        return data_out

def fourmension(data):
    num_sample = data.shape[0]
    num_row = 2
    num_column = data.shape[1]

    data_iq = np.zeros([num_sample, num_row, num_column, 1])
    for i in range(num_sample):
        data_iq[i,0,:,0] = np.real(data[i])
        data_iq[i,1,:,0] = np.imag(data[i])

    data_iq = np.transpose(data_iq, (0, 3, 2, 1))

    return data_iq

class Contras_Dataset_rec(Dataset):
    def __init__(self, data_input, label_input, rec_input, reset_prob=0.1):
        self.data_input = data_input
        self.label_input = label_input
        self.rec_input = rec_input
        self.selected_indices = set() 
        self.reset_prob = reset_prob
        
    def reset_selected_indices(self):
        self.selected_indices.clear()

    def __len__(self):
        return len(self.data_input)

    def __getitem__(self, index):
        if random.random() < self.reset_prob:
            self.selected_indices.clear()
        data_input = self.data_input[index]
        sender_label = self.label_input[index]
        receiver_label = self.rec_input[index]

        pos_index = self._find_positive_pair(sender_label, receiver_label)
        if pos_index is None:
            data_input_pos = data_input
        else:
            data_input_pos = self.data_input[pos_index]
            self.selected_indices.add(pos_index)  

        data_agu1 = self._process_data(data_input)
        data_agu2 = self._process_data(data_input_pos)
        label_input = sender_label.astype(float)
        # rx_input = receiver_label.astype(float)
        
        # view_spec(data_agu1[0])
        # print(data_agu1)
        
        # return data_agu1, data_agu2, label_input, rx_input
        return data_agu1, data_agu2, label_input

    def _find_positive_pair(self, sender_label, receiver_label):
        for i in range(len(self.data_input)):
            if (np.array_equal(self.label_input[i], sender_label) and 
                not np.array_equal(self.rec_input[i], receiver_label) ):
                
                return i
        return None

    def _process_data(self, data_in):
        data_out = data_aug_operator(data_in, multipath=random.randint(0, 1))
        # data_out = data_aug_operator(data_in, multipath=False)
        data_out = normalization(data_out)
        data_out = spectrogram(data_out[0], win_len=256, crop_ratio=0.3)
        # data_out = channel_ind_spectrogram(data_out[0], win_len=256, crop_ratio=0.3)
        data_out = torch.from_numpy(data_out.astype(np.float32))
        return data_out
    

def to_onehot(label_in):

        u, label_int = np.unique(label_in, return_inverse=True)
        num_classes = len(u)
        label_one_hot = np.eye(num_classes, dtype='uint8')[label_int]

        return label_one_hot, num_classes

def signal_preprocess_for_cnn(data):
    data = torch.FloatTensor(data)
    # print(data.shape)
    #  data for CNN

    data = data.permute(0, 3, 1, 2)
    
    # data_resqueezed = data.squeeze(-1).unsqueeze(1)
    # print(data.shape)
    return data