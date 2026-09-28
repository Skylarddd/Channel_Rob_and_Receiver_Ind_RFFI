import numpy as np
from scipy import signal
import math
import numpy as np
import h5py
from numpy import sum, sqrt
from numpy.random import standard_normal, uniform
from scipy import signal
from scipy.fftpack import dct


def normalization(data):
    ''' Normalize the signal.'''
    s_norm = np.zeros(data.shape, dtype=complex)

    for i in range(data.shape[0]):

        sig_amplitude = np.abs(data[i])
        rms = np.sqrt(np.mean(sig_amplitude**2))
        s_norm[i] = data[i]/rms

    return s_norm

def spectrogram(data, win_len, crop_ratio):

    data = normalization(data)
    overlap = round(0.5*win_len)
    num_sample = data.shape[0]

    num_row = len(range(math.floor(win_len*crop_ratio),math.ceil(win_len*(1-crop_ratio))))
    num_column = int(np.floor((data.shape[1]-win_len)/(win_len - overlap)) + 1)
    data_spec = np.zeros([num_sample, num_row, num_column, 1])

    for i in range(data.shape[0]):

        f, t, spec = signal.stft(data[i],
                                window='boxcar',
                                nperseg = win_len,
                                noverlap = overlap,
                                nfft = win_len,
                                return_onesided=False,
                                padded = False,
                                boundary = None)
        # print(spec.shape)

        spec = np.fft.fftshift(spec, axes=0)
        spec = spec_crop(spec, crop_ratio)
        spec = np.log10(np.abs(spec+1e-12)**2)

        data_spec[i,:,:,0] = spec
        # print(data_spec.shape)

    return data_spec


def spec_crop(x, crop_ratio):

    num_row = x.shape[0]
    x_cropped = x[math.floor(num_row*crop_ratio):math.ceil(num_row*(1-crop_ratio))]

    return x_cropped



def dspectrogram(data, win_len, crop_ratio):
    data = normalization(data)

    overlap = round(0.5*win_len)

    num_sample = data.shape[0]
    # num_row = math.ceil(win_len*(1-2*crop_ratio))
    num_row = len(range(math.floor(win_len*crop_ratio),math.ceil(win_len*(1-crop_ratio))))
    num_column = int(np.floor((data.shape[1]-win_len)/(win_len - overlap)) + 1) - 1

    data_dspec = np.zeros([num_sample, num_row, num_column, 1])
    # data_dspec = []
    for i in range(num_sample):
        dspec_amp = gen_differential_spectrogram(data[i], win_len, overlap)
        dspec_amp = spec_crop(dspec_amp, crop_ratio)
        data_dspec[i,:,:,0] = dspec_amp
        # # data_dspec[i,:,:,1] = dspec_phase

    return data_dspec


def gen_differential_spectrogram(sig, win_len, overlap):
    f, t, spec = signal.stft(sig,
                            window='boxcar',
                            nperseg= win_len,
                            noverlap= overlap,
                            nfft= win_len,
                            return_onesided=False,
                            padded = False,
                            boundary = None)

    # spec = spec_shift(spec)
    spec = np.fft.fftshift(spec, axes=0)
    # spec = spec_crop(spec, crop_ratio)

    # dspec = np.zeros([spec.shape[0],spec.shape[1]-1], dtype = complex)
    # for j in range(dspec.shape[1]):
    #     dspec[:,j] = spec[:,j] / spec[:,j+1]

    dspec = spec[:,1:]/(spec[:,:-1]+1e-32)

    dspec_amp = np.log10((np.abs(dspec)**2+1e-32))
    # dspec_phase = np.angle(dspec)

    return dspec_amp
