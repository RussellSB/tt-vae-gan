import soundfile as sf
from tqdm import tqdm
from io import StringIO
import numpy as np
import librosa
import shutil
import os

# Constants
WAVPATH = '../../ebagan-voice-conversion/database/spkr_2' # the output directory
SPK = 7
MAX = 100000

# This just gets the filenames with speaker referenes (because it is not clear in the filename otherwise)
path = '../flickr_audio/flickr_audio/wav2spk.txt'
s = StringIO(u"1,1.3,abcde")
speaker_files = np.genfromtxt(path, dtype=[('mystring','S27'),('myint','i8')], delimiter=' ')

# Saves amnt samples belonging to speaker from list of speaker files
def flickr_prep_wavs(outdir, speaker_files, src, max_amnt):
    
    print('Extracting and copying all source files to specified directory...')
    if not os.path.exists(outdir): os.mkdir(outdir)
    
    amnt = 0
    for filename, spk in speaker_files:
        if amnt >= max_amnt: break
        
        if spk == src: 
            # I would just need to change this part for URMP
            f = '../flickr_audio/flickr_audio/wavs/' + filename.decode()   
            shutil.copy(f, WAVPATH)
            
    print('Finished!')
        
flickr_prep_wavs(WAVPATH, speaker_files, SPK, MAX)