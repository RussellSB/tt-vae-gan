import soundfile as sf
from tqdm import tqdm
from io import StringIO
import numpy as np
import librosa
import os

# Constants
WAVPATH = '../pool/wavs-fl-4/' # the output directory
SPK = 4
MAX = 100000

# This just gets the filenames with speaker referenes (because it is not clear in the filename otherwise)
path = '../flickr_audio/flickr_audio/wav2spk.txt'
s = StringIO(u"1,1.3,abcde")
speaker_files = np.genfromtxt(path, dtype=[('mystring','S27'),('myint','i8')], delimiter=' ')

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
                outfilename = outdir + filename + '_' + f"{i:02}" + '.wav'   
                sf.write(outfilename, sub_wav, sr) 
                if end > full_samples: break

# Saves amnt samples belonging to speaker from list of speaker files
# Also subsambled to 1.6s / 128 frames. But named in a way that it can later be concatenated
def flickr_prep_wavs(outdir, speaker_files, src, max_amnt):
    
    print('Extracting and subdividing all source files to specified directory...')
    if not os.path.exists(outdir): os.mkdir(outdir)
    
    amnt = 0
    for filename, spk in speaker_files:
        if amnt >= max_amnt: break
        
        if spk == src: 
            # I would just need to change this part for URMP
            f = '../flickr_audio/flickr_audio/wavs/' + filename.decode()   
            x, sr = librosa.load(os.path.join(f), sr=16000)  # loads audio with librosa and downsamples it
            filename = filename.decode().split('.')[0]  # removing .wav from end
            subsample_x(x, outdir, filename, 128, 200, 16000)
            amnt += 1
            
    print('Finished!')
        
flickr_prep_wavs(WAVPATH, speaker_files, SPK, MAX)