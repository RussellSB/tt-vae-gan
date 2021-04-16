# voice-conversion-gan (Work in Progress)
A neural style transfer model for speech.

The VAE-CYCLE-GAN is based on [Ehab A. AlBadawy](https://ebadawy.github.io/post/speech_style_transfer/). The model is currently experiencing training instability issues. It suffers from mode collapse, and seemingly Gaussian blurred output. Trying to overcome it with a variety of tweaking. It may either be due to including an extra convolutional layer in both the encoder and decoder. Or something wrong with the loss function definitions.
