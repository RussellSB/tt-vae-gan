import numpy as np
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from utils import load_pickle, save_pickle, show_mel, show_mel_transfer
import itertools
import torch
import os

from models import Encoder, ResGen, Generator
from io import StringIO
import skimage.metrics

g = '0'  # gpu setting
device = torch.device('cuda')
torch.cuda.set_device(int(g))
map_location='cuda:'+g

n = '71'  # experiment out pool
path = '../pool/'+n


def to_numpy(data):
    mel = data.data[0]
    mel = mel.view(128, 128)
    mel = mel.detach().cpu().numpy()
    return mel


def forward_A2B(mel_in):
    with torch.no_grad():
    
        real_mel_A = torch.from_numpy(np.array([mel_in])).to(device)
        real_mel_A = real_mel_A.view(1, 1, 128, 128) # using batch size of 1 to return each metric

        # Forward pass for A to B
        latent_mel_A, mu_A, logvar_A = enc(real_mel_A)
        pseudo_mel_A = res(latent_mel_A)
        fake_mel_B = dec_A2B(pseudo_mel_A)

        # Cyclic reconstuction from fake B to A
        latent_recon_B, mu_recon_B, logvar_recon_B = enc(fake_mel_B)
        pseudo_recon_B = res(latent_recon_B)
        recon_mel_A = dec_B2A(pseudo_recon_B) 
    
    return to_numpy(real_mel_A), to_numpy(recon_mel_A), to_numpy(fake_mel_B)


def forward_B2A(mel_in):
    with torch.no_grad():
        
        real_mel_B = torch.from_numpy(np.array([mel_in])).to(device)
        real_mel_B = real_mel_B.view(1, 1, 128, 128) # using batch size of 1 to return each metric

        # Forward pass for B to A
        latent_mel_B, mu_B, logvar_B = enc(real_mel_B)
        pseudo_mel_B = res(latent_mel_B)
        fake_mel_A = dec_B2A(pseudo_mel_B)

        # Cyclic reconstuction from fake A to B
        latent_recon_A, mu_recon_A, logvar_recon_A = enc(fake_mel_A)
        pseudo_recon_A = res(latent_recon_A)
        recon_mel_B = dec_A2B(pseudo_recon_A) 
    
    return to_numpy(real_mel_B), to_numpy(recon_mel_B), to_numpy(fake_mel_A)


def compute_psnr_ssim(real_mel, recon_mel):
    # data range set to 2 as its distance between normalised min -1 and max 1
    psnr = skimage.metrics.peak_signal_noise_ratio(real_mel, recon_mel, data_range=2)
    ssim = skimage.metrics.structural_similarity(real_mel, recon_mel, data_range=2)
    
    return psnr, ssim


def eval_B2A(wavmels_B):
    psnr = []
    ssim = []
    
    # Holds numpy original wavs and style transfered mels
    path_gen = '../WAVENET-VOCODER/egs/gaussian/dump/lj/logmelspectrogram/norm/'+n+'_B2A/'
    if not os.path.exists(path_gen): os.makedirs(path_gen)
        
    # Holds mel spectro image outputs, and .npy metric arrays
    path_pool= path+'/test_B2A/'
    if not os.path.exists(path_pool): os.makedirs(path_pool)
    
    traintxt = '../WAVENET-VOCODER/egs/gaussian/dump/lj/logmelspectrogram/norm/'+n+'_B2A/train.txt'
    open(traintxt, 'w').close() # Clears file from any previous runs
    f = open(traintxt, 'a')  # Opens file for appending
    
    for i, wavmel in tqdm(enumerate(wavmels_B)):
        mel = np.load(test_path_B+wavmel['mel'].decode())
        real_mel_B, recon_mel_B, fake_mel_A = forward_B2A(mel) # B2A_logvar if logvar
        
        # Save generated mel for wavenet evaluation
        melstr = 'fake_mel_A_'+str(i)+'.npy'
        np.save(path_gen+melstr, fake_mel_A)

        # Save original wav for wavenet evaluation
        wavstr = wavmel['wav'].decode()
        wavrandom = np.load(test_path_B+wavstr)
        np.save(path_gen+wavstr, wavrandom) 

        f.write(wavstr+'|'+melstr+'|128|dummy\n')  # for train.txt in wavenet
        
        # Save image of the test case
        show_mel_transfer(real_mel_B, recon_mel_B, fake_mel_A, path_pool+'a_fake_'+str(i)+'.png')
        
        # Compute psnr and ssim
        p, s = compute_psnr_ssim(real_mel_B, recon_mel_B)
        psnr.append(p)
        ssim.append(s)
        
    np.save(path_pool+'psnr.npy',np.array(psnr))
    np.save(path_pool+'ssim.npy',np.array(ssim))
    
    print('==== Evaluation Metrics ====')
    print('Average PSNR: ', np.mean(psnr))
    print('Average SSIM: ', np.mean(ssim))
        
    return


def eval_A2B(wavemels_A):
    psnr = []
    ssim = []
    
    # Holds numpy original wavs and style transfered mels
    path_gen = '../WAVENET-VOCODER/egs/gaussian/dump/lj/logmelspectrogram/norm/'+n+'_A2B/'
    if not os.path.exists(path_gen): os.makedirs(path_gen)
        
    # Holds mel spectro image outputs, and .npy metric arrays
    path_pool= path+'/test_A2B/'
    if not os.path.exists(path_pool): os.makedirs(path_pool)
    
    traintxt = '../WAVENET-VOCODER/egs/gaussian/dump/lj/logmelspectrogram/norm/'+n+'_A2B/train.txt'
    open(traintxt, 'w').close() # Clears file from any previous runs
    f = open(traintxt, 'a')  # Opens file for appending
    
    for i, wavmel in tqdm(enumerate(wavmels_A)):
        mel = np.load(test_path_A+wavmel['mel'].decode())
        real_mel_A, recon_mel_A, fake_mel_B = forward_A2B(mel)  # A2B_logvar if logvar

        # Save generated mel for wavenet evaluation
        melstr = 'fake_mel_B_'+str(i)+'.npy'
        np.save(path_gen+melstr, fake_mel_B)

        # Save original wav also for wavenet evalutation
        wavstr = wavmel['wav'].decode()
        wavrandom = np.load(test_path_A+wavstr)
        np.save(path_gen+wavstr, wavrandom) 
            
        f.write(wavstr+'|'+melstr+'|128|dummy\n')  # for train.txt in wavenet
        
        # Save image of the test case
        show_mel_transfer(real_mel_A, recon_mel_A, fake_mel_B, path_pool+'b_fake_'+str(i)+'.png')
        
        # Compute psnr and ssim
        p, s = compute_psnr_ssim(real_mel_A, recon_mel_A)
        psnr.append(p)
        ssim.append(s)
        
    np.save(path_pool+'psnr.npy',np.array(psnr))
    np.save(path_pool+'ssim.npy',np.array(ssim))
    
    print('==== Evaluation Metrics ====')
    print('Average PSNR: ', np.mean(psnr))
    print('Average SSIM: ', np.mean(ssim))
        

if __name__ == "__main__":
    # Model Instantiation
    enc = Encoder().to(device)  # Shared encoder model
    res = ResGen().to(device)  # Shared residual decoding block
    dec_A2B = Generator().to(device)  # Generator and Discriminator for Speaker A to B
    dec_B2A = Generator().to(device)  # Generator and Discriminator for Speaker B to A

    # Initialise pretrained weights
    enc.load_state_dict(torch.load(path+'/enc.pt', map_location=map_location)) 
    res.load_state_dict(torch.load(path+'/res.pt', map_location=map_location)) 
    dec_A2B.load_state_dict(torch.load(path+'/dec_A2B.pt', map_location=map_location))
    dec_B2A.load_state_dict(torch.load(path+'/dec_B2A.pt', map_location=map_location))

    enc.eval()
    res.eval()
    dec_A2B.eval()
    dec_B2A.eval()

    # Load the paths
    test_path_A = '../WAVENET-VOCODER/egs/gaussian/dump/lj/logmelspectrogram/norm/eval/'
    wavmels_A = np.genfromtxt(test_path_A+'train.txt', dtype=[('wav','S50'),('mel','S50'),('nmel','i8'),('str','S27')], delimiter='|')

    # Only temporary
    test_path_B = '../WAVENET-VOCODER/egs/gaussian/dump/lj/logmelspectrogram/norm/eval_4/'
    wavmels_B = np.genfromtxt(test_path_B+'train.txt', dtype=[('wav','S50'),('mel','S50'),('nmel','i8'),('str','S27')], delimiter='|')
    
    eval_A2B(wavmels_A)
    eval_B2A(wavmels_B)