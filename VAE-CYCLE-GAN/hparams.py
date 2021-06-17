# id for reference
n = '93' # 3 conv, 5 res basic blocks  
g = 1

# Settings related to dataset (as saved from wavenet preprocessing)
datadir = 'lj' # 'tt-2' or 'lj'
if datadir == 'tt-2':
    A, B = 'vn', 'tpt' 
    max_duplets = 1352
if datadir == 'lj':
    A, B = '7', '4' 
    max_duplets = 1680 

# Training settings
curr_epoch = 0
max_epochs = 100
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
lambda_structure = 100.0 # 100.0 (this is an original addition)

# Model architecture parameters
num_res = 3  # number of resnet blocks for resnet phases (recommended minimum is 3)
res_type = 'basic'  # choose either ['basic', 'bottleneck']