from tqdm import tqdm
import shutil
import glob
import os

dataroot = '../../datasets/urmp/'
outdir = '../data/data_urmp/'

# Gets the audio seperated wavs of a specific instrument from the URMP dataset
def get_audiosep_ins(ins):
    return glob.glob(dataroot + '**/AuSep*'+ins+'*.wav', recursive = True)

# Saves wavs belonging to speaker from list of speaker files
def urmp_prep_wavs(outdir, instrument_files, src):
    os.makedirs(outdir, exist_ok=True)
    for f in tqdm(instrument_files, desc="extracting ins %s"%src):
        shutil.copy(f, outdir)
        
trumpet_files = get_audiosep_ins('tpt')
violin_files = get_audiosep_ins('vn')

urmp_prep_wavs(outdir+'spkr_1', trumpet_files, 'tpt')
urmp_prep_wavs(outdir+'spkr_2', violin_files, 'vn')