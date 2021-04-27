# vc-vae-cycle-gan (Work in Progress)
A neural style transfer model for speech, able to transfer the voice of one speaker to the voice of another (vc for voice conversion). This implementation is based on [Ehab A. AlBadawy's work](https://ebadawy.github.io/post/speech_style_transfer/).

Due to deadlines, this repo isn't in the most representable of states right now - but the model learns! Error rates do minimise with time. For better learning consider training with more epochs than 100, or simply adding more consequetive residual necks in the encoder in model.py files. This was not done due to resource limitations from my end. Nonetheless, please refer to my code in VAE-CYCLE-GAN. Feel free to use the architecture from model.py and refer to training.py for forward inference and loss definitions. 

The current architecture only works with melspectrograms of a 128x128 size. Melspectrogram parameters are the same as that of the paper. Updates for user-friendliness and adabtability to come in the future.

## TODO
- Fork WaveNet repo and commit changes
- Organise preprocessing stage for easy replication
- Organise test/evaluation stage for easy replication
- Link dataset
- Update readme with setup tutorial

## NOTES
- VAE loss without unit variance versions are in its own sperate scripts for now (_logvar)
- Can choose either 'bce', 'mse', or Wasserstenian mode for adversarial loss
- Discriminator will use sigmoid at end only if BCE (like DCGAN)
- No fancy data loader is used for now, data is loaded as a list of images
