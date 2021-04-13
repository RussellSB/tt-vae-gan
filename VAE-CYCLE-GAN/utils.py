import pickle
import librosa
import librosa.display
from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings( "ignore", module = "matplotlib\..*" )  # ignore plot warnings

import torch
import torch.nn as nn
import numpy as np

import random

def save_pickle(obj, path):
    with open(path, 'wb') as fp:
        pickle.dump(obj, fp)


def load_pickle(path):
    with open(path, 'rb') as fp:
        return pickle.load(fp)

# Plots mels that are already amplitude scaled
def show_mel(mel):
    plt.figure(figsize=(10, 2))
    img = librosa.display.specshow(mel, x_axis='time', y_axis='mel', sr=16000, hop_length=200)
    plt.colorbar(format="%+2.f dB")

# Plots mels that are already amplitude scaled
def show_mel_transfer(mel_in, mel_out, save_path, mel_s=False):
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
    
    if not mel_s: mel_in = librosa.power_to_db(mel_in, ref=np.max)
    im = librosa.display.specshow(mel_in, x_axis='time', y_axis='mel', sr=16000, hop_length=200, ax=ax[0])
    ax[0].set(title='Input Melspectrogram')
    ax[0].label_outer()
    
    if not mel_s: mel_out = librosa.power_to_db(mel_out, ref=np.max)
    im = librosa.display.specshow(mel_out, x_axis='time', y_axis='mel', sr=16000, hop_length=200, ax=ax[1])
    ax[1].set(title='Output Melspectrogram')
    ax[1].label_outer()
    
    plt.savefig(save_path)

# Stores generated output from past 50 iterations 
# (Author: https://github.com/Lornatang/CycleGAN-PyTorch/blob/master/cyclegan_pytorch/utils.py)
class ReplayBuffer:
    def __init__(self, max_size=10):
        assert (max_size > 0), "Empty buffer or trying to create a black hole. Be careful."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.cat(to_return)

# Initializes weights
# (Author: https://github.com/Lornatang/CycleGAN-PyTorch/blob/master/cyclegan_pytorch/utils.py)
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)