import torch
device = 'cuda' # torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(1)

import numpy as np
from tqdm.auto import tqdm
import itertools

from utils import load_pickle, save_pickle, ReplayBuffer, weights_init, show_mel
from models import ResidualBlock, Encoder, Generator, Discriminator

from matplotlib import pyplot as plt
import librosa

# Hyperparameters
max_epochs = 100
max_duplets = 3000
batch_size = 4
learning_rate = 0.0001
assert max_duplets % batch_size == 0, 'Max sample pairs must be divisible by batch size!' 

# Regularisation
lambda_cycle = 1 #0.001
lambda_enc = 1 #100.0
lambda_dec = 1 #10.0

# Loading training data
melset_7_128 = load_pickle('pool/melset_7_128_100.pickle')  # for now loading subset for testing
melset_4_128 = load_pickle('pool/melset_4_128_100.pickle')  # for now loading subset for testing
print('Melset A size:', len(melset_7_128), 'Melset B size:', len(melset_4_128))

# Shuffling melspectrograms
rng = np.random.default_rng()
melset_7_128 = rng.permutation(np.array(melset_7_128))
melset_4_128 = rng.permutation(np.array(melset_4_128))
melset_7_128 = torch.from_numpy(melset_7_128)  # Torch conversion
melset_4_128 = torch.from_numpy(melset_4_128)

# Model Instantiation
enc = Encoder().to(device)  # Shared encoder model (with partial decoding)
dec_A2B = Generator().to(device)  # Generator and Discriminator for Speaker A to B
disc_B = Discriminator().to(device)
dec_B2A = Generator().to(device)  # Generator and Discriminator for Speaker B to A
disc_A = Discriminator().to(device)

enc.apply(weights_init)  # Initialise weights
dec_A2B.apply(weights_init)
dec_B2A.apply(weights_init)
disc_A.apply(weights_init)
disc_B.apply(weights_init)

fake_A_buffer = ReplayBuffer()  # Instantiate buffers
fake_B_buffer = ReplayBuffer()

optim_enc = torch.optim.Adam(enc.parameters(), lr=learning_rate)  # Initialise optimizers
optim_dec = torch.optim.Adam(itertools.chain(dec_A2B.parameters(), dec_B2A.parameters()),lr=learning_rate)
optim_disc_A = torch.optim.Adam(disc_A.parameters(), lr=learning_rate)
optim_disc_B = torch.optim.Adam(disc_B.parameters(), lr=learning_rate)

loss_adversarial = torch.nn.BCELoss().to(device)  # Initialize criterions
loss_cycle = torch.nn.L1Loss().to(device)

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


pbar = tqdm(range(max_epochs), desc='Epochs')  # init epoch pbar
for i in pbar:
    
    pbar_sub = tqdm(range(0, max_duplets, batch_size),leave=False, desc='Batches')  # init batch pbar
    for j in pbar_sub:
        
        # Loading real samples from each speaker in batches
        #real_mel_A = melset_7_128[j : j + batch_size].to(device)
        #real_mel_B = melset_4_128[j : j + batch_size].to(device)
        
	# Testing that loss can firstly go down with same batch
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
        fake_mel_A = dec_B2A(latent_mel_B)
        fake_output_A = torch.squeeze(disc_A(fake_mel_A))
        # Cyclic reconstuction from fake A to B
        latent_fake_A, mu_fake_A, logvar_fake_A = enc(fake_mel_A)
        recon_mel_B = dec_A2B(latent_fake_A)  
        
        # Forward pass for A to B
        latent_mel_A, mu_A, logvar_A = enc(real_mel_A)
        fake_mel_B = dec_A2B(latent_mel_A)
        fake_output_B = torch.squeeze(disc_B(fake_mel_B))
        # Cyclic reconstuction from fake B to A
        latent_fake_B, mu_fake_B, logvar_fake_B = enc(fake_mel_B)
        recon_mel_A = dec_B2A(latent_fake_B)  
        
        # Encoding loss A and B
        kld_A = 0 #-0.5 * torch.sum(1 + logvar_A - mu_A.pow(2) - logvar_A.exp())
        mse_A = (recon_mel_A - real_mel_A).pow(2).mean()
        loss_enc_A = (kld_A + mse_A) * lambda_enc
        
        kld_B = 0 #-0.5 * torch.sum(1 + logvar_B - mu_B.pow(2) - logvar_B.exp())
        mse_B = (recon_mel_B - real_mel_B).pow(2).mean()
        loss_enc_B = (kld_B + mse_B) * lambda_enc
        
        # Decoder/Generator loss
        loss_dec_B2A = loss_adversarial(fake_output_A, real_label) * lambda_dec
        loss_dec_A2B = loss_adversarial(fake_output_B, real_label) * lambda_dec
        
        # Cyclic loss
        loss_cycle_ABA = loss_cycle(recon_mel_A, real_mel_A) * lambda_cycle
        loss_cycle_BAB = loss_cycle(recon_mel_B, real_mel_B) * lambda_cycle
        
        # Backward pass for generator and update all  generators
        errDec = loss_dec_A2B + loss_dec_B2A + loss_cycle_ABA + loss_cycle_BAB#loss_dec_A2B + loss_dec_B2A + loss_cycle_ABA + loss_cycle_BAB + loss_enc_A + loss_enc_B
        errDec.backward()
        optim_enc.step()
        optim_dec.step()
        
        # =====================================================
        #                   Discriminators update
        # =====================================================
        
        # Resetting gradients
        optim_disc_A.zero_grad()
        optim_disc_B.zero_grad()
        
        # Forward pass disc_A
        real_out_A = torch.squeeze(disc_A(real_mel_A))
        fake_mel_A = fake_A_buffer.push_and_pop(fake_mel_A)
        fake_out_A = torch.squeeze(disc_A(fake_mel_A.detach()))
        
        loss_D_real_A = loss_adversarial(real_out_A, real_label)
        loss_D_fake_A = loss_adversarial(fake_out_A, fake_label)
        errDisc_A = (loss_D_real_A + loss_D_fake_A) / 2
        
        # Forward pass disc_B
        real_out_B = torch.squeeze(disc_B(real_mel_B))
        fake_mel_B = fake_B_buffer.push_and_pop(fake_mel_B)
        fake_out_B = torch.squeeze(disc_B(fake_mel_B.detach()))

        loss_D_real_B = loss_adversarial(real_out_B, real_label)
        loss_D_fake_B = loss_adversarial(fake_out_B, fake_label)
        errDisc_B = (loss_D_real_B + loss_D_fake_B) / 2
        
        # Backward pass and update all
        errDisc_A.backward()
        errDisc_B.backward()
        optim_disc_A.step()
        optim_disc_B.step() 
        
        # Update error log
        pbar.set_postfix(vA=loss_enc_A.item(),vB=loss_enc_B.item(), A2B=loss_dec_A2B.item(), B2A=loss_dec_B2A.item(), 
        ABA=loss_cycle_ABA.item(), BAB=loss_cycle_BAB.item(), disc_A=errDisc_A.item(), disc_B=errDisc_B.item())
        
    	# Update error history    
        train_hist['enc_A'].append(loss_enc_A.item())
        train_hist['enc_B'].append(loss_enc_B.item())
        train_hist['dec_B2A'].append(loss_dec_B2A)
        train_hist['dec_A2B'].append(loss_dec_A2B)
        train_hist['dec_ABA'].append(loss_cycle_ABA)
        train_hist['dec_BAB'].append(loss_cycle_BAB)
        train_hist['dec'].append(errDec.item())
        train_hist['disc_A'].append(errDisc_A.item())
        train_hist['disc_B'].append(errDisc_B.item())    

    # Saving updated training history and model weights every 10 epochs
    if(i % 10 == 0):
        save_pickle(train_hist, 'pool/01/train_hist.pickle')
        torch.save(dec_A2B.state_dict(), 'pool/01/dec_A2B.pt')
        torch.save(dec_B2A.state_dict(), 'pool/01/dec_B2A.pt')
        torch.save(enc.state_dict(), 'pool/01/enc.pt')
        torch.save(disc_A.state_dict(), 'pool/01/disc_A.pt')
        torch.save(disc_B.state_dict(), 'pool/01/disc_B.pt')

    # Save generated output every epoch
    save_pickle(fake_A_buffer, 'pool/01/a/a_fake_epoch_'+str(i)+'.pickle')
    save_pickle(fake_B_buffer, 'pool/01/b/b_fake_epoch_'+str(i)+'.pickle')
