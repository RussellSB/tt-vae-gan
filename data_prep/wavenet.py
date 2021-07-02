from tqdm import tqdm
import numpy as np
import librosa
import argparse
import librosa
import pickle
import os

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, help="path to prepared dataset folder")
parser.add_argument("--outdir", type=str, help="path to output directory")
args = parser.parse_args()
print(args)
    
print('Loading spectrogram data...')
feats_train = pickle.load(open('%s/data_train.pickle'%(args.dataset),'rb'))
feats_eval = pickle.load(open('%s/data_eval.pickle'%(args.dataset),'rb'))
feats_test = pickle.load(open('%s/data_test.pickle'%(args.dataset),'rb'))

print('Loading corresponding filename references...')
refs_train = pickle.load(open('%s/refs_train.pickle'%(args.dataset),'rb'))
refs_eval = pickle.load(open('%s/refs_eval.pickle'%(args.dataset),'rb'))
refs_test = pickle.load(open('%s/refs_test.pickle'%(args.dataset),'rb'))

def to_wavenet(feats_spk, refs_spk, wavdir, outdir):   
    
    os.makedirs(outdir, exist_ok=True)
    
    f_txt = '%s/train.txt' % outdir
    open(f_txt, 'w').close() # clears from previous runs
    txt = open(f_txt,"w+") # opens txt for appending
    
    # iterate through and save spect and audio
    for spect, fname in tqdm(zip(feats_spk, refs_spk), total=len(refs_spk), desc='extracting to %s'%outdir):
        
        # Load audio based on fname
        f = os.path.join(wavdir, fname)
        audio, _ = librosa.load(f, sr=16000)
        
        # Prepare file names to save 
        name = fname[:-4]  # removes .wav from audio fname
        f_spect = name + '-feats.npy'
        f_audio = name + '-wave.npy'
        
        # Append line of filenames in train.txt
        n_frames = spect.shape[1]
        txt.write('%s|%s|%d|dummy\n' % (f_spect, f_audio, n_frames))
        
        # Save files
        np.save(os.path.join(outdir, f_spect), spect)
        np.save(os.path.join(outdir, f_audio), audio)

# Data preparation
num_spk = len(feats_train)
for i in range(num_spk):
    outdir = os.path.join(args.outdir, 'spkr_%d'%(i+1))
    wavdir = os.path.join(args.dataset, 'spkr_%d'%(i+1))
    
    to_wavenet(feats_train[i], refs_train[i], wavdir, os.path.join(outdir, 'train'))
    to_wavenet(feats_eval[i], refs_eval[i], wavdir, os.path.join(outdir, 'eval'))
    to_wavenet(feats_test[i], refs_test[i], wavdir, os.path.join(outdir, 'test'))
