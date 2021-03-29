import pickle
import librosa
import librosa.display
from matplotlib import pyplot as plt
import numpy as np

def save_pickle(obj, path):
    with open(path, 'wb') as fp:
        pickle.dump(obj, fp)
        
def load_pickle(path):
    with open(path, 'rb') as fp:
        return pickle.load(fp)

def show_mel(mel):
    plt.figure(figsize=(10, 2))
    S_dB = librosa.power_to_db(mel, ref=np.max)
    img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=16000, hop_length=200)
    plt.colorbar(format="%+2.f dB")
