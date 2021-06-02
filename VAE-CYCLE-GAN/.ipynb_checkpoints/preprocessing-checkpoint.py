import numpy as np
from tqdm.auto import tqdm
from utils import load_pickle, save_pickle, show_mel
from io import StringIO
import os

# Path of preprocessed wavenet spectrograms. Formats into pickle to load at once by this project
path = '../WAVENET/egs/gaussian/dump/lj/logmelspectrogram/norm/train_no_dev/'

def cont_wn_to_arr(speaker):
    melset = []
    
    # Iterate through wavmel paths
    for i, wavmel in tqdm(enumerate(wavmels)):
        ref = wavmel[1].decode()
        
        # Skip some inconsistencies from wavenet preprocess
        if ref == 'dummy': 
            continue
            
        spk_curr = ref.split('_')[1]
        
        if spk_curr == speaker:
            mel = np.load(path+ref)
            if(mel.shape[0] == 128 and mel.shape[1] == 128): melset.append(mel)
            
    return melset


wavmels = np.genfromtxt(path+'train.txt', dtype=[('wav','S50'),('mel','S50'),('nmel','i8'),('str','S27')], delimiter='|')
melset_7_cont_wn = cont_wn_to_arr('7')
melset_4_cont_wn = cont_wn_to_arr('4')

save_pickle(melset_7_cont_wn, '../pool/melset_7_128_cont_wn.pickle')
save_pickle(melset_4_cont_wn, '../pool/melset_4_128_cont_wn.pickle')