# Timbre Transfer with VAE-GAN & WaveNet

This pipeline follows and extends the work of [Albadawy & Lyu 2020](https://ebadawy.github.io/post/speech_style_transfer/Albadawy_et_al-2020-INTERSPEECH.pdf). The work I used this for shows (amongst other things) that their proposed **voice** conversion model is also applicable to context of musical instruments, therefore reforming the conversion to a more generalised audio style - **timbre**.

## Summary
The implemented pipeline makes use of the following submodules (click for orginating repos):
1. [voice_conversion](https://github.com/ebadawy/voice_conversion) - Performs VAE-GAN style transfer in the time-frequency melspectrogram domain.
2. [wavenet_vocoder](https://github.com/r9y9/wavenet_vocoder) - Vocodes melspectrogram output from style transfer model to realistic audio.
3. [fad](https://github.com/google-research/google-research/tree/master/frechet_audio_distance) - Computes Fréchet Audio Distance (using VGGish) to evaluate the quality of wavenet vocoder output.

## Demo

 *(Audio coming soon)*

Male to Female:

![m1_t_f1](https://user-images.githubusercontent.com/35470600/127534797-2b6d8f30-3072-49ee-921f-86a9f7174a2a.png)

Violin to Trumpet: 

![vn_t_tpt](https://user-images.githubusercontent.com/35470600/127534790-75bfed70-64be-497d-9d87-7ec26e59ea7f.png)

## Requirements
Recommended GPU VRAM per model:
- voice_conversion - **2 GB**
- wavenet_vocoder - **8 GB**
- fad - **16 GB** 

### Note
- If fad exceeds your computational resources, you can ignore it. It is not necessary for timbre transfer - only for evaluating it.
- If wavenet_vocoder exceeds your resources, you can try a less computationally intense vocoder (such as [melgan](https://github.com/seungwonpark/melgan))

## Setup

Clone this repo as well as the sub modules for *voice_conversion* and *wavenet_vocoder* with git:

```
git clone https://github.com/RussellSB/tt-vae-gan.git
cd tt-vae-gan 
git submodule init 
git submodule update
```

# Tutorial

Below is a brief tutorial on how you can use the submodules in conjunction with eachother. This is the pipeline I followed for my project. For more information - feel free to refer to the documentation in the originating repos, and raise an issue here or in *voice_conversion* if something is unclear / does not work.

## 1. VAE-GAN
## 2. WaveNet
## 3. FAD
