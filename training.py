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
lambda_cycle = 0.001
lambda_VAE = 100.0
lambda_GAN = 10.0

# Loading training data
melset_7_128 = load_pickle('pool/melset_7_128.pickle')
melset_4_128 = load_pickle('pool/melset_4_128.pickle')
print('Melset A size:', len(melset_7_128), 'Melset B size:', len(melset_4_128))

# Shuffling melspectrograms
rng = np.random.default_rng()
melset_7_128 = rng.permutation(np.array(melset_7_128))
melset_4_128 = rng.permutation(np.array(melset_4_128))
melset_7_128 = torch.from_numpy(melset_7_128)  # Torch conversion
melset_4_128 = torch.from_numpy(melset_4_128)

# Model Instantiation
E = Encoder().to(device)  # Shared encoder model (with partial decoding)
G_A2B = Generator().to(device)  # Generator and Discriminator for Speaker A to B
D_B = Discriminator().to(device)
G_B2A = Generator().to(device)  # Generator and Discriminator for Speaker B to A
D_A = Discriminator().to(device)

E.apply(weights_init)  # Initialise weights
G_A2B.apply(weights_init)
G_B2A.apply(weights_init)
D_A.apply(weights_init)
D_B.apply(weights_init)

fake_A_buffer = ReplayBuffer()  # Instantiate buffers
fake_B_buffer = ReplayBuffer()

optim_E = torch.optim.Adam(E.parameters(), lr=learning_rate)  # Initialise optimizers
optim_G = torch.optim.Adam(itertools.chain(G_A2B.parameters(), G_B2A.parameters()),lr=learning_rate)
optim_D_A = torch.optim.Adam(D_A.parameters(), lr=learning_rate)
optim_D_B = torch.optim.Adam(D_B.parameters(), lr=learning_rate)

loss_adversarial = torch.nn.MSELoss().to(device)  # Initialize criterions
loss_cycle = torch.nn.L1Loss().to(device)

train_hist = {}  # Initialise loss history lists
train_hist['G_B2A'] = []
train_hist['G_ABA'] = []
train_hist['G_A2B'] = [] 
train_hist['G_BAB'] = [] 
train_hist['G'] = []
train_hist['D_A'] = []
train_hist['D_B'] = []
train_hist['VAE_A'] = []
train_hist['VAE_B'] = []

pbar = tqdm(range(max_epochs), desc='Epochs')  # init epoch pbar
for i in pbar:
    
    pbar_sub = tqdm(range(0, max_duplets, batch_size),leave=False, desc='Batches')  # init batch pbar
    for j in pbar_sub:
        
        # Loading real samples from each speaker in batches
        real_mel_A = melset_7_128[j : j + batch_size].to(device)
        real_mel_B = melset_4_128[j : j + batch_size].to(device)
        
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
        optim_E.zero_grad()
        optim_G.zero_grad()   

        # Forward pass for B to A
        latent_mel_B, mu_B, logvar_B = E(real_mel_B)
        fake_mel_A = G_B2A(latent_mel_B)
        fake_output_A = torch.squeeze(D_A(fake_mel_A))
        
        # Cyclic reconstuction from fake A to B
        latent_fake_A, mu_fake_A, logvar_fake_A = E(fake_mel_A)
        recon_mel_B = G_A2B(latent_fake_A)  
        
        # Forward pass for A to B
        latent_mel_A, mu_A, logvar_A = E(real_mel_A)
        fake_mel_B = G_A2B(latent_mel_A)
        fake_output_B = torch.squeeze(D_B(fake_mel_B))
        
        # Cyclic reconstuction from fake B to A
        latent_fake_B, mu_fake_B, logvar_fake_B = E(fake_mel_B)
        recon_mel_A = G_B2A(latent_fake_B)  
        
        # Encoding loss A and B
        kld_A = -0.5 * torch.sum(1 + logvar_A - mu_A.pow(2) - logvar_A.exp())
        mse_A = (recon_mel_A - real_mel_A).pow(2).mean()
        loss_VAE_A = (kld_A + mse_A) * lambda_VAE
        
        kld_B = -0.5 * torch.sum(1 + logvar_B - mu_B.pow(2) - logvar_B.exp())
        mse_B = (recon_mel_B - real_mel_B).pow(2).mean()
        loss_VAE_B = (kld_B + mse_B) * lambda_VAE
        
        errVAE = (loss_VAE_A + loss_VAE_B) / 2
        errVAE.backward(retain_graph=True)  # retain graph so other losses can update in same graph
        optim_E.step()
        
        # GAN loss
        loss_GAN_B2A = loss_adversarial(fake_output_A, real_label) * lambda_GAN
        loss_GAN_A2B = loss_adversarial(fake_output_B, real_label) * lambda_GAN
        
        # Cyclic loss
        loss_cycle_ABA = loss_cycle(recon_mel_A, real_mel_A) * lambda_cycle
        loss_cycle_BAB = loss_cycle(recon_mel_B, real_mel_B) * lambda_cycle
        
        # Backward pass and update all  
        errG = loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
        errG.backward()
        optim_G.step()
        
        # =====================================================
        #                   Discriminators update
        # =====================================================
        
        # Resetting gradients
        optim_D_A.zero_grad()
        optim_D_B.zero_grad()
        
        # Forward pass D_A
        real_out_A = torch.squeeze(D_A(real_mel_A))
        fake_mel_A = fake_A_buffer.push_and_pop(fake_mel_A)
        fake_out_A = torch.squeeze(D_A(fake_mel_A.detach()))
        
        loss_D_real_A = loss_adversarial(real_out_A, real_label)
        loss_D_fake_A = loss_adversarial(fake_out_A, fake_label)
        errD_A = (loss_D_real_A + loss_D_fake_A) / 2
        
        # Forward pass D_B
        real_out_B = torch.squeeze(D_B(real_mel_B))
        fake_mel_B = fake_B_buffer.push_and_pop(fake_mel_B)
        fake_out_B = torch.squeeze(D_B(fake_mel_B.detach()))

        loss_D_real_B = loss_adversarial(real_out_B, real_label)
        loss_D_fake_B = loss_adversarial(fake_out_B, fake_label)
        errD_B = (loss_D_real_B + loss_D_fake_B) / 2
        
        # Backward pass and update all
        errD_A.backward()
        errD_B.backward()
        optim_D_A.step()
        optim_D_B.step() 
        
        # Update error log
        pbar.set_postfix(errVAE=errVAE.item(), errG=errG.item(), lossGA2B=loss_GAN_A2B.item(), lossGB2A=loss_GAN_B2A.item(), 
        lossGABA=loss_cycle_ABA.item(), lossGBAB=loss_cycle_BAB.item(), errD_A=errD_A.item(), errD_B=errD_B.item())
        
    # Update error epoch history    
    train_hist['VAE_A'].append(loss_VAE_A.item())
    train_hist['VAE_B'].append(loss_VAE_B.item())
    
    train_hist['G_B2A'].append(loss_GAN_B2A)
    train_hist['G_A2B'].append(loss_GAN_A2B)
    train_hist['G_ABA'].append(loss_cycle_ABA)
    train_hist['G_BAB'].append(loss_cycle_BAB)
    train_hist['G'].append(errG.item())
    
    train_hist['D_A'].append(errD_A.item())
    train_hist['D_B'].append(errD_B.item())    

    # Saving updated training history and model weights every 10 epochs
    if(i % 10 == 0):
        save_pickle(train_hist, 'pool/01/train_hist.pickle')
        torch.save(G_A2B.state_dict(), 'pool/01/G_A2B.pt')
        torch.save(G_B2A.state_dict(), 'pool/01/G_B2A.pt')
        torch.save(E.state_dict(), 'pool/01/E.pt')
        torch.save(D_A.state_dict(), 'pool/01/D_A.pt')
        torch.save(D_B.state_dict(), 'pool/01/D_B.pt')

    # Save generated output every epoch
    save_pickle(fake_A_buffer, 'pool/01/a_fake_epoch_'+i+'.pickle')
    save_pickle(fake_B_buffer, 'pool/01/b_fake_epoch_'+i+'.pickle')