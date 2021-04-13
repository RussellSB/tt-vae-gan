import torch
device = 'cuda' # torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(1)

import numpy as np
from tqdm.auto import tqdm
import itertools

from utils import load_pickle, save_pickle, ReplayBuffer, weights_init, show_mel, show_mel_transfer
from models import Encoder, ResGen, Generator, Discriminator

from matplotlib import pyplot as plt
import librosa

import os

# Prepares result output
n = '24' # modified moidel, two res blocks, wasserstenien w/ lower clipping
print('Outputting to pool', n)
pooldir = '../pool/' + str(n)
adir = pooldir + '/a'
bdir = pooldir + '/b'

# If folder doesn't exist make it
if not os.path.exists(pooldir):
    os.mkdir(pooldir)
else:
    print("Warning: Outputing to an existing experiment pool!", n)
    
if not os.path.exists(adir):
    os.mkdir(adir)
if not os.path.exists(bdir):
    os.mkdir(bdir)

# Hyperparameters
max_epochs = 100
max_duplets = 3000 #1680 #5940
batch_size = 4
learning_rate = 0.0001
clip_value = 0.001 # lower and upper clip value for discriminator weights
assert max_duplets % batch_size == 0, 'Max sample pairs must be divisible by batch size!' 

# Loss weighting
lambda_cycle = 1 #1/100 #100.0
lambda_enc = 1 #1/100 #100.0
lambda_dec = 1 #10.0
lambda_kld = 1 #0.001
lambda_latent = 1 #10.0

# Loading training data
melset_7_128 = load_pickle('../pool/melset_7_128_cont.pickle') 
melset_4_128 = load_pickle('../pool/melset_4_128_cont.pickle')
print('Melset A size:', len(melset_7_128), 'Melset B size:', len(melset_4_128))
print('Max duplets:', max_duplets)

# Shuffling melspectrograms
rng = np.random.default_rng()
melset_7_128 = rng.permutation(np.array(melset_7_128))
melset_4_128 = rng.permutation(np.array(melset_4_128))
melset_7_128 = torch.from_numpy(melset_7_128)  # Torch conversion
melset_4_128 = torch.from_numpy(melset_4_128)

# Model Instantiation
enc = Encoder().to(device)  # Shared encoder model
res = ResGen().to(device)  # Shared residual decoding block
dec_A2B = Generator().to(device)  # Generator and Discriminator for Speaker A to B
disc_B = Discriminator().to(device)
dec_B2A = Generator().to(device)  # Generator and Discriminator for Speaker B to A
disc_A = Discriminator().to(device)

# Initialise weights
enc.apply(weights_init) 
res.apply(weights_init)  
dec_A2B.apply(weights_init)
dec_B2A.apply(weights_init)
disc_A.apply(weights_init)
disc_B.apply(weights_init)

# Instantiate buffers
fake_A_buffer = ReplayBuffer()  
real_A_buffer = ReplayBuffer()  
fake_B_buffer = ReplayBuffer()
real_B_buffer = ReplayBuffer()

# Initialise optimizers
optim_enc = torch.optim.Adam(enc.parameters(), lr=learning_rate) 
optim_res = torch.optim.Adam(enc.parameters(), lr=learning_rate) 
optim_dec = torch.optim.Adam(itertools.chain(dec_A2B.parameters(), dec_B2A.parameters()),lr=learning_rate)
optim_disc_A = torch.optim.Adam(disc_A.parameters(), lr=learning_rate)
optim_disc_B = torch.optim.Adam(disc_B.parameters(), lr=learning_rate)

train_hist = {}  # Initialise loss history lists
train_hist['dec_B2A'] = []
train_hist['dec_ABA'] = []
train_hist['dec_A2B'] = [] 
train_hist['dec_BAB'] = [] 
train_hist['dec'] = []
train_hist['disc_A'] = []
train_hist['disc_B'] = []
train_hist['enc_A'] = []
train_hist['enc_B'] = []
train_hist['enc_lat'] = []

# =====================================================================================================
#                                       Loss functions
# =====================================================================================================

# Initialize criterions
criterion_latent = torch.nn.L1Loss().to(device)
criterion_adversarial = torch.nn.BCELoss().to(device)  

# Encoder loss function for encoder, tries to retain some degree of information
def loss_encoding(logvar, mu, fake_mel, real_mel):
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    mse = (fake_mel - real_mel).pow(2).mean()
    return ((kld * lambda_kld) + mse) * lambda_enc

# Cyclic loss for reconstruction through opposing encoder, tries not to retain degree of info too closely
def loss_cycle(logvar, mu, recon_mel, real_mel):
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    mse = (recon_mel - real_mel).pow(2).mean()
    return ((kld * lambda_kld) + mse) * lambda_cycle

# Latent loss, L1 distance between centroids of each speaker's distribution
def loss_latent(mu_A, mu_B):
    return criterion_latent(mu_A, mu_B) * lambda_latent

# Adversarial loss function for decoder and discriminator seperately
def loss_adversarial(output, label, dec=True):
    loss = criterion_adversarial(output, label) * lambda_dec
    if(dec): loss *= lambda_dec
    return loss

# =====================================================================================================
#                                       The Training Loop
# =====================================================================================================
pbar = tqdm(range(max_epochs), desc='Epochs')  # init epoch pbar
for i in pbar:
    
    pbar_sub = tqdm(range(0, max_duplets, batch_size),leave=False, desc='Batches')  # init batch pbar
    for j in pbar_sub:
        
        # Loading real samples from each speaker in batches
#         real_mel_A = melset_7_128[j : j + batch_size].to(device)
#         real_mel_B = melset_4_128[j : j + batch_size].to(device)
        
	    #Testing that loss can firstly go down with same batch
        real_mel_A = melset_7_128[0 : batch_size].to(device)
        real_mel_B = melset_4_128[0 : batch_size].to(device)
        
        # Resizing to model tensors
        real_mel_A = real_mel_A.view(batch_size, 1, 128, 128)
        real_mel_B = real_mel_B.view(batch_size, 1, 128, 128)

        # Real data labelled 1, fake data labelled 0
        batch_size = real_mel_A.size(0)
        real_label = torch.squeeze(torch.full((batch_size, 1), 1, device=device, dtype=torch.float32))
        fake_label = torch.squeeze(torch.full((batch_size, 1), 0, device=device, dtype=torch.float32))
        
        # =====================================================
        #            Encoders and Decoding Generators update
        # =====================================================

        # Resetting gradients
        optim_enc.zero_grad()
        optim_dec.zero_grad()   

        # Forward pass for B to A
        latent_mel_B, mu_B, logvar_B = enc(real_mel_B)
        pseudo_mel_B = res(latent_mel_B)
        fake_mel_A = dec_B2A(pseudo_mel_B)
        fake_output_A = torch.squeeze(disc_A(fake_mel_A))
        
        # Cyclic reconstuction from fake A to B
        latent_fake_A, mu_fake_A, logvar_fake_A = enc(fake_mel_A)
        pseudo_fake_A = res(latent_fake_A)
        recon_mel_B = dec_A2B(pseudo_fake_A)  
        
        # Forward pass for A to B
        latent_mel_A, mu_A, logvar_A = enc(real_mel_A)
        pseudo_mel_A = res(latent_mel_A)
        fake_mel_B = dec_A2B(pseudo_mel_A)
        fake_output_B = torch.squeeze(disc_B(fake_mel_B))
        
        # Cyclic reconstuction from fake B to A
        latent_fake_B, mu_fake_B, logvar_fake_B = enc(fake_mel_B)
        pseudo_fake_B = res(latent_fake_B)
        recon_mel_A = dec_B2A(pseudo_fake_B)  
        
        # Encoding loss A and B
        loss_enc_A = loss_encoding(logvar_A, mu_A, fake_mel_B, real_mel_A)
        loss_enc_B = loss_encoding(logvar_B, mu_B, fake_mel_A, real_mel_B)
        
        # Decoder/Generator loss
        loss_dec_B2A = -torch.mean(fake_output_A) # loss_adversarial(fake_output_A, real_label) #
        loss_dec_A2B = -torch.mean(fake_output_B) # loss_adversarial(fake_output_B, real_label) #
        
        # Cyclic loss
        loss_cycle_ABA = loss_cycle(logvar_A, mu_A, recon_mel_A, real_mel_A)
        loss_cycle_BAB = loss_cycle(logvar_B, mu_B, recon_mel_B, real_mel_B)

        # Latent loss
        loss_lat = loss_latent(mu_A, mu_B)
        
        # Backward pass for encoder and update all res/dec generator components
        errDec = loss_dec_A2B + loss_dec_B2A + loss_cycle_ABA + loss_cycle_BAB + loss_enc_B + loss_enc_A + loss_lat 
        errDec.backward()
        optim_enc.step()
        optim_res.step()
        optim_dec.step()
        
        # =====================================================
        #                   Discriminators update
        # =====================================================
        
        # Resetting gradients
        optim_disc_A.zero_grad()
        optim_disc_B.zero_grad()
        
        # Forward pass disc_A
        real_out_A = torch.squeeze(disc_A(real_mel_A))
        real_B_buffer.push_and_pop(real_mel_B)  # Add real to buffer
        fake_mel_A = fake_A_buffer.push_and_pop(fake_mel_A)
        fake_out_A = torch.squeeze(disc_A(fake_mel_A.detach()))
        
        loss_D_real_A = -torch.mean(real_out_A) #loss_adversarial(real_out_A, real_label) #
        loss_D_fake_A = torch.mean(fake_out_A) #loss_adversarial(fake_out_A, fake_label) #
        errDisc_A = loss_D_real_A + loss_D_fake_A #(loss_D_real_A + loss_D_fake_A) / 2
        
        # Forward pass disc_B
        real_out_B = torch.squeeze(disc_B(real_mel_B))
        real_A_buffer.push_and_pop(real_mel_A)  # Add real to buffer
        fake_mel_B = fake_B_buffer.push_and_pop(fake_mel_B)
        fake_out_B = torch.squeeze(disc_B(fake_mel_B.detach()))

        loss_D_real_B = -torch.mean(real_out_B) # loss_adversarial(real_out_B, real_label, dec=False) #
        loss_D_fake_B = torch.mean(fake_out_B) # loss_adversarial(fake_out_B, fake_label, dec=False) #
        errDisc_B = loss_D_real_B + loss_D_fake_B #(loss_D_real_B + loss_D_fake_B) / 2
        
        # Backward pass and update all
        errDisc_A.backward()
        errDisc_B.backward()
        optim_disc_A.step()
        optim_disc_B.step() 
        
        # Clip discriminator parameters
        for p in disc_A.parameters():
            p.data.clamp_(-clip_value, clip_value)
            
        for p in disc_B.parameters():
            p.data.clamp_(-clip_value, clip_value)
        
        # Update error log
        pbar.set_postfix(vA=loss_enc_A.item(),vB=loss_enc_B.item(), A2B=loss_dec_A2B.item(), B2A=loss_dec_B2A.item(), 
        ABA=loss_cycle_ABA.item(), BAB=loss_cycle_BAB.item(), disc_A=errDisc_A.item(), disc_B=errDisc_B.item())
        
    # Update error history every epoch 
    train_hist['enc_A'].append(loss_enc_A.item())
    train_hist['enc_B'].append(loss_enc_B.item())
    train_hist['enc_lat'].append(loss_lat.item())
    train_hist['dec_B2A'].append(loss_dec_B2A.item())
    train_hist['dec_A2B'].append(loss_dec_A2B.item())
    train_hist['dec_ABA'].append(loss_cycle_ABA.item())
    train_hist['dec_BAB'].append(loss_cycle_BAB.item())
    train_hist['dec'].append(errDec.item())
    train_hist['disc_A'].append(errDisc_A.item())
    train_hist['disc_B'].append(errDisc_B.item())    

    # Saving updated training history and model weights every 10 epochs
    if(i % 10 == 0 or i == 99):
        save_pickle(train_hist, pooldir +'/train_hist.pickle')
        torch.save(dec_A2B.state_dict(),  pooldir +'/dec_A2B.pt')
        torch.save(dec_B2A.state_dict(),  pooldir +'/dec_B2A.pt')
        torch.save(enc.state_dict(), pooldir +'/enc.pt')
        torch.save(disc_A.state_dict(),  pooldir +'/disc_A.pt')
        torch.save(disc_B.state_dict(),  pooldir +'/disc_B.pt')

    # Save generator B2A output per epoch
    d_in, d_out = real_B_buffer.data[0], fake_A_buffer.data[0]
    mel_in, mel_out = torch.squeeze(d_in).cpu().numpy(), torch.squeeze(d_out).cpu().numpy()
    show_mel_transfer(mel_in, mel_out, pooldir + '/a/a_fake_epoch_'+ str(i) + '.png')
    
    # Save generator A2B output per epoch
    d_in, d_out = real_A_buffer.data[0], fake_B_buffer.data[0]
    mel_in, mel_out = torch.squeeze(d_in).cpu().numpy(), torch.squeeze(d_out).cpu().numpy()
    show_mel_transfer(mel_in, mel_out, pooldir + '/b/b_fake_epoch_'+ str(i) + '.png')
    
    