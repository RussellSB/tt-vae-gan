import librosa
import argparse
import pickle
import os

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, help="path to prepared dataset folder")
parser.add_argument("--outdir", type=str, help="path to output directory")
args = parser.parse_args()
print(args)

#os.makedirs(args.outdir, exist_ok=True)
    
print('Loading spectrogram data...')
feats_train = pickle.load(open('%s/data_train.pickle'%(args.dataset),'rb'))
feats_eval = pickle.load(open('%s/data_eval.pickle'%(args.dataset),'rb'))
feats_test = pickle.load(open('%s/data_test.pickle'%(args.dataset),'rb'))

print('Loading corresponding filename references...')
refs_train = pickle.load(open('%s/refs_train.pickle'%(args.dataset),'rb'))
refs_eval = pickle.load(open('%s/refs_eval.pickle'%(args.dataset),'rb'))
refs_test = pickle.load(open('%s/refs_test.pickle'%(args.dataset),'rb'))

def to_wavenet(feats_spk, refs_spk, wavdir, outdir):    
    for spect, fname in zip(feats_spk, refs_spk):
        f = os.path.join(wavdir, fname)
        print(f)

# Data preparation
num_spk = len(feats_train)
for i in range(num_spk):
    outdir = os.path.join(args.outdir, 'spk_%d'%(i+1))
    wavdir = os.path.join(args.dataset, 'spk_%d'%(i+1))
    
    to_wavenet(feats_train[i], refs_train[i], wavdir, os.path.join(outdir, 'train'))
    to_wavenet(feats_train[i], refs_train[i], wavdir, os.path.join(outdir, 'eval'))
    to_wavenet(feats_train[i], refs_train[i], wavdir, os.path.join(outdir, 'test'))
