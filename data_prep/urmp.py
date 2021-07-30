from tqdm import tqdm
import argparse
import shutil
import glob
import os

parser = argparse.ArgumentParser()
parser.add_argument("--dataroot", type=str, default='../../datasets/urmp/', help="data root of flickr")
parser.add_argument("--outdir", type=str, default='../voice_conversion/data/data_urmp/', help="output directory")
args = parser.parse_args()
print(args)

# Gets the audio seperated wavs of a specific instrument from the URMP dataset
def get_audiosep_ins(ins):
    return glob.glob(args.dataroot + '**/AuSep*'+ins+'*.wav', recursive = True)

# Saves wavs belonging to speaker from list of speaker files
def urmp_prep_wavs(outdir, instrument_files, src):
    os.makedirs(outdir, exist_ok=True)
    for f in tqdm(instrument_files, desc="extracting ins %s"%src):
        shutil.copy(f, outdir)
        
trumpet_files = get_audiosep_ins('tpt')
violin_files = get_audiosep_ins('vn')

# Data preparation
urmp_prep_wavs(args.outdir+'spkr_1', trumpet_files, 'tpt')
urmp_prep_wavs(args.outdir+'spkr_2', violin_files, 'vn')
#urmp_prep_wavs(args.outdir+'spkr_3', flute_files, 'fl')  # uncomment to include flute
#urmp_prep_wavs(args.outdir+'spkr_4', cello_files, 'vc')  # uncomment to include cello