# vc-vae-cycle-gan (Work in Progress)

![Architecture](https://github.com/RussellSB/vc-vae-cycle-gan/blob/main/images/VC-VAE-CYCLE-GAN.JPG)

A neural style transfer model for speech, able to transfer the voice of one speaker to the voice of another (vc for voice conversion). This implementation is based on [Ehab A. AlBadawy's work](https://ebadawy.github.io/post/speech_style_transfer/).

The current architecture only works with melspectrograms of a 128x128 size. Melspectrogram parameters are the same as that of the paper. Updates for user-friendliness and adabtability to come in the future. Results are currently best with using an MSE advarserial loss, and they aren't yet as great as the paper's results but are interpretable, and are currently in the process of improvement. 

The ultimate scope of this project is to achieve replicable results, as well as also extend it to instrument timbre transfer for comparison purposes (voice conversion but in the context of instruments).

## Showcase

Albeit not at the same high quality as AlBadawy, it achieves interpretable conversions:
![Voice conversion from Male to Female](https://github.com/RussellSB/vc-vae-cycle-gan/blob/main/images/a2b.png) ![Voice conversion from Female to Male](https://github.com/RussellSB/vc-vae-cycle-gan/blob/main/images/b2a.png)

*Left: Male to Female. As can be seen the formants become more distant expanding over a higher pitch range.  Right: Female to Male. Here, the formants get compressed to a lower pitch range. Note that for both of these results, Output is what is forward infered and Reconstructed is what is infered backwards from the generated output. This follows the logistics of CycleGAN (inferring fake samples cyclically). These result visualisations are what was best achieved to date - using an MSE adversarial loss.*

## Resources
- [WaveNet Vocoder](https://github.com/r9y9/wavenet_vocoder)
- [UNIT Repository](https://github.com/mingyuliutw/UNIT)
- [Flickr Audio Corpus Dataset](https://groups.csail.mit.edu/sls/downloads/flickraudio/)


## Brief Tutorial
1. First download the Flickr Audio Corpus Dataset
2. Preprocess voice samples of the two speakers of interest using the implemented melspectrogram methodology of WaveNet
3. In VAE-CYCLE-GAN, run preprocess.py with the path set as the preprocessing output of WAVENET-VOCODER
4. Run training.py, and make any suitable changes in model.py to run within the current computational budget
5. After training from VAE-CYCLE-GAN, run evaluate.py to infer style transfer on unseen melspectrograms
6. Ensure that WAVENET-VOCODER is trained on the same training set as VAE-CYCLE-GAN (May follow its respective repo for this)
7. After this, from WAVENET-VOCODER run evaluate.py to infer audio from the generated melspectrograms

## Further Notes
- Can choose either 'bce', 'mse', or Wasserstenian mode for adversarial loss (currently mse is best)
- In order to better fit a limited computational budget, can reduce number of residual blocks at the bottleneck in model.py
- Discriminator will use sigmoid at end only if BCE (like DCGAN)
- No data loader is used for now, data is loaded as a list of images with pickle
- Following WAVENET-VOCODER should be generally sufficient, in this project their gaussian egs experiment was relied on.
- Parameters for preprocessing melspectrograms were modified in gaussian wrt AlBadawy's work
- There is an issue with synthesis.py in WAVENET-VOCODER, instead you must use their evaluate.py in the following way:

```
python evaluate.py egs/gaussian/dump/lj/logmelspectrogram/norm/66_B2A/ \
    egs/gaussian/exp/lj_train_no_dev_main/checkpoint_step000349495.pth \
    out/66_B2A_generated 
```

Here, 66_B2A is the directory of the generated melspectrograms from the Cycle-GAN (test input to WaveNet). Meanwhile 66_B2A_generated is the directory of the corresponding generated audio wave files (test output of wavenet). 
