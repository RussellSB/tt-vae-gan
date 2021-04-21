# vc-vae-cycle-gan (Work in Progress)
A neural style transfer model for speech, able to transfer the voice of one speaker to the voice of another (vc for voice conversion). This implementation is based on [Ehab A. AlBadawy's work](https://ebadawy.github.io/post/speech_style_transfer/).

Due to dealines, this repo isn't in the most representable of states right now - but the model works! Please refer to my code in VAE-CYCLE-GAN. Feel free to use the architecture from model.py and refer to training.py for forward inference and loss definitions. 

The current architecture only works with melspectrograms of a 128x128 size. Melspectrogram parameters are the same as that of the paper. Updates for user-friendliness and adabtability to come in the future.

## TODO
- Fork WaveNet repo and commit changes
- Organise preprocessing stage for easy replication
- Organise test/evaluation stage for easy replication
- Link dataset
- Update readme with setup tutorial
