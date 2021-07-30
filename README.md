# Timbre Transfer with VAE-GAN & WaveNet

This pipeline follows and extends the work of [Albadawy & Lyu 2020](https://ebadawy.github.io/post/speech_style_transfer/Albadawy_et_al-2020-INTERSPEECH.pdf). The work I used this for shows (amongst other things) that their proposed **voice** conversion model is also applicable to context of musical instruments, therefore reforming the conversion to a more generalised audio style - **timbre**.

## Summary
The implemented pipeline makes use of the following projects (click for orginating repos):
1. [voice_conversion](https://github.com/ebadawy/voice_conversion) - Performs VAE-GAN style transfer in the time-frequency melspectrogram domain.
2. [wavenet_vocoder](https://github.com/r9y9/wavenet_vocoder) - Vocodes melspectrogram output from style transfer model to realistic audio.
3. [fad](https://github.com/google-research/google-research/tree/master/frechet_audio_distance) - Computes Fréchet Audio Distance (using VGGish) to evaluate the quality of wavenet vocoder output.

## Index
- [Demo](#demo)
- [Hardware](#hardware)
- [Tutorial](#tutorial)
- [Pretrained Models](#pretrained-models)

## Demo

 *(Audio coming soon)*

Male to Female:

![m1_t_f1](https://user-images.githubusercontent.com/35470600/127534797-2b6d8f30-3072-49ee-921f-86a9f7174a2a.png)

Violin to Trumpet: 

![vn_t_tpt](https://user-images.githubusercontent.com/35470600/127534790-75bfed70-64be-497d-9d87-7ec26e59ea7f.png)

## Hardware
Recommended GPU VRAM per model:
- voice_conversion - **2 GB**
- wavenet_vocoder - **8 GB**
- fad - **16 GB** 

### Note
- Keep in mind that if you train in a mnay to many context (ie more than 2 timbres) you may need more VRAM for *voice_conversion*
- If fad exceeds your computational resources, you can ignore it. It is not necessary for timbre transfer - only for evaluating it.
- If wavenet_vocoder exceeds your resources, you can try a less computationally intense vocoder (such as [melgan](https://github.com/seungwonpark/melgan))

## Tutorial

### 0. Setup

1. Clone this repo as well as its sub modules for *voice_conversion* and *wavenet_vocoder* with git:

```
git clone https://github.com/RussellSB/tt-vae-gan.git
cd tt-vae-gan 
git submodule init 
git submodule update
```

2. Ensure that your environment has installed the dependencies of the submodules that you will use.

### 1. VAE-GAN

#### 0. Download the dataset. 

Choose:

- Flickr 8k Audio for speakers ([link](https://groups.csail.mit.edu/sls/downloads/flickraudio/))
- URMP for instruments ([link](http://www2.ece.rochester.edu/projects/air/projects/URMP.html))


#### 1. Prepare your data. 

Run one of the python commands for extracting timbre files of interest:

```
cd data_prep
python flickr --dataroot [path/to/flickr_audio/flickr_audio/]  # For Flickr
python urmp --dataroot [path/to/urmp/]  # For URMP
```

- By default this will output to ```voice_conversion/data/data_[name]/```. 
- You can add more timbres by duplicating lines 27-28 and changing each last argument to the timbre id of interest.

#### 2. Preprocess your data



3. Train on your data.

4. Infer with VAE-GAN and reconstruct raw audio with Griffin Lim.

### 2. WaveNet

1. Prepare your data again (based on data extracted for VAE-GAN).

2. Preprocess your data again (based on WaveNet specs this time).

3. Train a wavenet vocoder per timbre.

4. Infer with style transferred Griffin Lim reconstructions as input to improve their perceptual quality.

### 3. FAD

0. Download the VGGish model pretrained on AudioSet.

1. Use it to embed real training data and estimate multivariate Gaussians.

2. Use it to embed generated test data (post-WaveNet) and estimate multivariate Gaussians.

3. Compute FAD between stats of the real and generated.

## Pretrained Models

*Coming Soon*
