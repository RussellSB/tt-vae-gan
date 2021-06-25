# id for reference
n = '110'
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
curr_epoch = 100
max_epochs = 200
batch_size = 4 # 4, 8
learning_rate = 0.0001

# Adversarial loss function
loss_mode = 'mse'  # set to 'bce' or 'mse'

# Loss function weighting 
lambda_cycle = 0 # 100.0 # 0
lambda_enc = 100.0 # 100.0 
lambda_dec = 10.0 #10.0
lambda_kld = 0.001  # 0.001
lambda_latent = 10.0 # 10.0
#lambda_structure = 100.0 # 100.0 (this is an original addition)

# Model architecture parameters
num_res = 3  # number of resnet blocks for resnet phases (recommended minimum is 3)
res_type = 'basic'  # choose either ['basic', 'bottleneck']

# Evaluate config
infer_type = 'short'  #  long, short