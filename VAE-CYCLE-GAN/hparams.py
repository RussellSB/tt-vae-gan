# id for reference
n = '78'  # Residual bottleneck fixed, with dimensional difference (stride=1 for conv 3x3 layer)
g = 1  # Graphics card number

# Training settings
curr_epoch = 0
max_epochs = 100
max_duplets = 1680 
batch_size = 8 # 4
learning_rate = 0.0001

# Adversarial loss function
loss_mode = 'mse'  # set to 'bce' or 'mse' or 'ws'
isWass = False # either true or false to make a wGAN (negates loss_mode when True)
clip_value = 0.0001 # lower and upper clip value for discriminator weights (used when isWass is True)

# Loss function weighting 
lambda_cycle = 100.0 # 100.0 
lambda_enc = 100.0 # 100.0 
lambda_dec = 10.0 #10.0 # 10.0 # 1.0
lambda_kld = 0.0001
lambda_latent = 10.0 # 10.0