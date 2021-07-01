from tqdm import tqdm
import numpy as np
import shutil
import os

dataroot = '../../datasets/flickr_audio/flickr_audio/'
outdir = '../data/data_flickr/'

# This just gets the filenames with speaker referenes (because it is not clear in the filename otherwise)
textin = dataroot + 'wav2spk.txt'
speaker_files = np.genfromtxt(textin, dtype=[('mystring','S27'),('myint','i8')], delimiter=' ')

# Saves wavs belonging to speaker from list of speaker files
def flickr_prep_wavs(outdir, speaker_files, src):
    os.makedirs(outdir, exist_ok=True)
    
    files = [filename for (filename, spk) in speaker_files if spk == src]
    for filename in tqdm(files, desc="extracting spk %s"%src):
        f = dataroot + 'wavs/' + filename.decode()   
        shutil.copy(f, outdir)

# Preperation for preprocessing from VAEGAN repo
flickr_prep_wavs(outdir+'spkr_1', speaker_files, 4)
flickr_prep_wavs(outdir+'spkr_2', speaker_files, 7)