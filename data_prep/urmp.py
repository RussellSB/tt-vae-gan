import soundfile as sf
from tqdm import tqdm
import numpy as np
import librosa
import shutil
import glob
import os

# Constants
WAVPATH = '../../ebagan-voice-conversion/db_urmp/spkr_1' # the output directory
INS = 'tpt'
MAX = 100000

# Gets the audio seperated wavs of a specific instrument from the URMP dataset
# Can set ins as 'vc' or 'tpt'
def get_audiosep_ins(ins):
    return glob.glob('../urmp/**/AuSep*'+ins+'*.wav', recursive = True)

# Returns subsamples from audiosample of frame length 128 (in the case of 16000 sr, 800 n_fft, 200 hop length) 
# Accepts audio x, saves its subsamples to outdir
def subsample_x(x, outdir, filename, goal_frames, hop, sr):

        #ipd.display(ipd.Audio(x, rate=sr))
        sub_samples = (goal_frames * hop) # the desired samples we want so they will have 128 frames each after processing
        full_samples = len(x)
        
        if full_samples >= sub_samples:
            
            for i, t in enumerate(range(0, full_samples, sub_samples)):
                start = t
                end = t + sub_samples
                
                # if out of bounds, pad with zeros
                if end > full_samples: 
                    sub_wav = np.concatenate([x[start:], np.zeros(end-full_samples-1)])
                else: # if in bounds just extract normally
                    sub_wav = x[start:end-1]
                    
                #if sub_wav.size != 25599:
                #   print(sub_wav.size)
                outfilename = outdir + filename + '_' + f"{i:03}" + '.wav'
                sf.write(outfilename, sub_wav, sr) 
                if end > full_samples: break

# Saves amnt samples belonging to speaker from list of speaker files
# Also subsambled to 1.6s / 128 frames. But named in a way that it can later be concatenated
def urmp_prep_wavs(outdir, instrument_files, src, max_amnt):
    
    print('Extracting and subdividing all source files to specified directory...')
    if not os.path.exists(outdir): os.mkdir(outdir)
    
    amnt = 0
    for f in instrument_files:
        if amnt >= max_amnt: break
        shutil.copy(f, WAVPATH)
        amnt += 1
            
    print('Finished!')
        
instrument_files = get_audiosep_ins(INS)
urmp_prep_wavs(WAVPATH, instrument_files, INS, MAX)