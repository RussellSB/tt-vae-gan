from tqdm import tqdm
import numpy as np
import argparse
import shutil
import os

parser = argparse.ArgumentParser()
parser.add_argument("--dataroot", type=str, default='../../datasets/flickr_audio/flickr_audio/', help="data root of flickr")
parser.add_argument("--outdir", type=str, default='../voice_conversion/data/data_flickr/', help="output directory")
args = parser.parse_args()
print(args)

# This just gets the filenames with speaker referenes (because it is not clear in the filename otherwise)
textin = args.dataroot + 'wav2spk.txt'
speaker_files = np.genfromtxt(textin, dtype=[('mystring','S27'),('myint','i8')], delimiter=' ')

# Saves wavs belonging to speaker from list of speaker files
def flickr_prep_wavs(outdir, speaker_files, src):
    os.makedirs(outdir, exist_ok=True)
    
    files = [filename for (filename, spk) in speaker_files if spk == src]
    for filename in tqdm(files, desc="extracting spk %s"%src):
        f = args.dataroot + 'wavs/' + filename.decode()   
        shutil.copy(f, outdir)

# Data preparation
flickr_prep_wavs(args.outdir+'spkr_1', speaker_files, 4)
flickr_prep_wavs(args.outdir+'spkr_2', speaker_files, 7)
#flickr_prep_wavs(args.outdir+'spkr_3', speaker_files, 49)  # uncomment to include another female
#flickr_prep_wavs(args.outdir+'spkr_4', speaker_files, 17)  # uncomment to include another male