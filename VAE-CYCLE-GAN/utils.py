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
    plt.figure()
    plt.imshow(np.rot90(mel), interpolation="None")
    plt.ylabel('Mels')
    plt.ylabel('Frames')
    plt.title('Melspectrogram')
    plt.tight_layout()

# Plots mels that are already amplitude scaled and saves
def show_mel_transfer(mel_in, mel_recon, mel_out, mel_target, save_path):
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(6, 6))
    
    ax[0,0].imshow(np.rot90(mel_in), interpolation="None")
    ax[0,0].set(title='Input')
    ax[0,0].set_ylabel('Mels')
    ax[0,0].axes.xaxis.set_ticks([])
    ax[0,0].axes.xaxis.set_ticks([])
    
    ax[1,0].imshow(np.rot90(mel_recon), interpolation="None")
    ax[1,0].set(title='Reconstructed Input')
    ax[1,0].set_xlabel('Frames')
    ax[1,0].set_ylabel('Mels')

    ax[0,1].imshow(np.rot90(mel_out), interpolation="None")
    ax[0,1].set(title='Output')
    ax[0,1].axes.yaxis.set_ticks([])
    ax[0,1].axes.xaxis.set_ticks([])
    
    ax[1,1].imshow(np.rot90(mel_target), interpolation="None")
    ax[1,1].set(title='Target')
    ax[1,1].set_xlabel('Frames')
    ax[1,1].axes.yaxis.set_ticks([])

    plt.savefig(save_path)
    plt.close()

    
# Plots mels that are already amplitude scaled and saves
def show_mel_transfer_eval(mel_in, mel_recon, mel_out, save_path):
    fig, ax = plt.subplots(nrows=1, ncols=3, sharex=True)
    
    ax[0].imshow(np.rot90(mel_in), interpolation="None")
    ax[0].set(title='Input')
    ax[0].set_ylabel('Mels')
    
    ax[1].imshow(np.rot90(mel_recon), interpolation="None")
    ax[1].set(title='Reconstructed Input')
    ax[1].axes.yaxis.set_ticks([])

    ax[2].imshow(np.rot90(mel_out), interpolation="None")
    ax[2].set(title='Output')
    ax[2].set_xlabel('Frames')
    ax[2].axes.yaxis.set_ticks([])

    plt.savefig(save_path)
    plt.close()
    

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