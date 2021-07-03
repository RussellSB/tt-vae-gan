from tqdm import tqdm
import argparse
import pickle
import shutil
import os

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, help="path to prepared dataset folder")
parser.add_argument("--outdir", type=str, help="path to output directory")
parser.add_argument("--tag", type=str, help="tag for datasource reference")
args = parser.parse_args()
print(args)

print('Loading corresponding filename references...')
refs_train = pickle.load(open('%s/refs_train.pickle'%(args.dataset),'rb'))
refs_eval = pickle.load(open('%s/refs_eval.pickle'%(args.dataset),'rb'))
refs_test = pickle.load(open('%s/refs_test.pickle'%(args.dataset),'rb'))

def to_wavenet(refs_spk, wavdir, outdir):   
    os.makedirs(outdir, exist_ok=True)
    for fname in tqdm(refs_spk, total=len(refs_spk), desc='extracting to %s'%outdir):
        f = os.path.join(wavdir, fname)
        shutil.copy(f, outdir)
        

# Data preparation
num_spk = len(refs_train)
for i in range(num_spk):
    outdir = os.path.join(args.outdir, args.tag + '_%d'%(i+1))
    wavdir = os.path.join(args.dataset, 'spkr_%d'%(i+1))
    
    to_wavenet(refs_train[i], wavdir, os.path.join(outdir, 'train_no_dev'))
    to_wavenet(refs_eval[i], wavdir, os.path.join(outdir, 'dev'))
    to_wavenet(refs_test[i], wavdir, os.path.join(outdir, 'eval'))
