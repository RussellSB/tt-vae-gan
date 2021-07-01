from tqdm import tqdm
import numpy as np
import argparse
import shutil
import os

parser = argparse.ArgumentParser()
parser.add_argument("--dataroot", type=str, default='../../datasets/flickr_audio/flickr_audio/', help="data root of flickr")
parser.add_argument("--outdir", type=str, default='../voice_conversion/data/data_flickr/', help="output directory")
opt = parser.parse_args()
print(opt)

# This just gets the filenames with speaker referenes (because it is not clear in the filename otherwise)
textin = opt.dataroot + 'wav2spk.txt'
speaker_files = np.genfromtxt(textin, dtype=[('mystring','S27'),('myint','i8')], delimiter=' ')

# Saves wavs belonging to speaker from list of speaker files
def flickr_prep_wavs(outdir, speaker_files, src):
    os.makedirs(outdir, exist_ok=True)
    
    files = [filename for (filename, spk) in speaker_files if spk == src]
    for filename in tqdm(files, desc="extracting spk %s"%src):
        f = opt.dataroot + 'wavs/' + filename.decode()   
        shutil.copy(f, outdir)

# Preperation for preprocessing from VAEGAN repo
flickr_prep_wavs(opt.outdir+'spkr_1', speaker_files, 4)
flickr_prep_wavs(opt.outdir+'spkr_2', speaker_files, 7)