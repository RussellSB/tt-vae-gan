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
- If you train many-to-many (more than 2 timbres) you may need a stronger GPU for *voice_conversion*
- If wavenet_vocoder exceeds your resources, you can try a less computationally intense vocoder (such as [melgan](https://github.com/seungwonpark/melgan))
- If fad exceeds your computational resources, you can ignore it. It is not necessary for timbre transfer - only for evaluating it.

## Tutorial

### 0. Setup

0.1. Clone this repo as well as its sub modules for *voice_conversion* and *wavenet_vocoder* with git:

```
git clone https://github.com/RussellSB/tt-vae-gan.git
cd tt-vae-gan 
git submodule init 
git submodule update
```

0.2. Ensure that your environment has installed the dependencies of the submodules.

---

### 1. VAE-GAN

#### 1.0. Download the dataset. 

Choose:

- Flickr 8k Audio for speakers ([link](https://groups.csail.mit.edu/sls/downloads/flickraudio/))
- URMP for instruments ([link](http://www2.ece.rochester.edu/projects/air/projects/URMP.html))
- Other


#### 1.1. Prepare your data. 

Run one of the python commands for extracting timbre files of interest:

```
cd data_prep
python flickr --dataroot [path/to/flickr_audio/flickr_audio/]  # For Flickr
python urmp --dataroot [path/to/urmp/]  # For URMP
```

- By default this will output to ```voice_conversion/data/data_[name]/```. 
- ```[name]``` would be either ```flickr``` or ```urmp```
- You can add more timbres by duplicating lines 27-28 and changing each last argument to the timbre id of interest.

Alternatively, you can use your own dataset. Just set it up so that in ```voice_conversion/data/data_mydataset``` you have the following structure:

```bash
voice_conversion/data/data_mydataset
├── spkr_1
│   ├── sample.wav
├── spkr_2
│   ├── sample.wav
│   ...
└── spkr_N
    ├── sample.wav
    ...
# The directory under each speaker cannot be nested.
```

#### 1.2. Preprocess your data

````
cd ../voice_conversion/src
python preprocess.py --dataset ../data/data_[name]
````

- Set more than two timbres by also adding ```--n_spkrs [int]```. By default ```n_spkrs=2```.

#### 1.3. Train on your data.

```
python train.py --model_name [expname] --dataset ../data/data_[name]
```

- Can set max epochs with ```--n_epochs [int]``` (100 default)
- Can set how often to save models with ```--checkpoint_interval [int]``` (1 epoch by default)

#### 1.4. Infer with VAE-GAN and reconstruct raw audio with Griffin Lim.

```
python inference.py --model_name [expname] --epoch [int] --trg_id 2 --src_id 1 --wavdir [path/to/testset_1]
```

- Instead of ```--wavdir``` you can do ```--wav``` for a single file input.
- Since only the data prep for wavenet creates audio directories for each train/eval/test split, use that.
- Do step 2.1. then come back to this. Can then set something like ```--wavdir ../../wavenet_vocoder/egs/gaussian/data/flickr_2/eval```.
- Griffin Lim reconstructions are low quality, but will be improved to sufficient realism when passed to WaveNet vocoder.

---

### 2. WaveNet

#### 2.1. Prepare your data again (based on data extracted for VAE-GAN).

```
cd ../../data_prep
python wavenet.py --dataset ../voice_conversion/data/data_[name] --outdir ../wavenet_vocoder/egs/gaussian/data --tag [name]
```

#### 2.2. Preprocess your data again (based on WaveNet specs this time).

```
cd ../wavenet_vocoder/egs/gaussian
spk="[name]_[id]" ./run.sh --stage 1 --stop-stage 1
```

- For two speakers ids this would be either ```1``` or ```2```.
- You need to run the .sh command for each target timbre.

#### 2.3. Train a wavenet vocoder.

```
spk="[name]_[id]" hparams=conf/[name].json ./run.sh --stage 2 --stop-stage 2 
```

- Just like preprocessing, you need to run this for each target timbre.
- You can add ```CUDA_VISIBLE_DEVICES="0,1"``` before ```./run.sh``` if you have two GPUs (training takes quite long).

#### 2.4. Infer style transferred reconstructions to improve their perceptual quality.

```
spk="[name]_[id_2]" inferdir="[expname]_[epoch]_G[id_2]_S[id_1]" hparams=conf/flickr.json ./infer.sh
```

- [id_2] is the target id. [id_1] is the source id.
- For example, for transfer from ids 1-to-2 with experiment 'initial' and trained VAE-GAN after epoch 99, ```inferdir="initial_99_G2_S1"```.
- You can also add ```CUDA_VISIBLE_DEVICES="0,1"``` before ```./infer.sh``` (inferring takes quite long).

---

### 3. FAD

#### 3.0. Download the VGGish model pretrained on AudioSet.

```
cd ../../../fad
mkdir -p data
curl -o data/vggish_model.ckpt https://storage.googleapis.com/audioset/vggish_model.ckpt
```

#### 3.1. Create csvs for referencing files of timbre sets (real train set, then fake test set, both of same target timbre)

```
ls --color=never ../wavenet_vocoder/egs/gaussian/data/[name]_[id_2]/train_no_dev/*.wav  > test_audio/[name]_[id_2].csv
ls --color=never ../wavenet_vocoder/egs/gaussian/out/[name]_[id_2]_[expname]_[epoch]_G[id_2]_S[id_1]/*_gen.wav > test_audio/[name]_[id_2]_[expname]_[epoch]_G[id_2]_S[id_1].csv
```

#### 3.2. Embed each of the timbre sets with VGGish

```
mkdir -p stats
python -m frechet_audio_distance.create_embeddings_main  --input_files test_audio/[name]_[id_2].csv \
                                                        --stats stats/[name]_[id_2]_stats
                                                      
python -m frechet_audio_distance.create_embeddings_main  --input_files test_audio/[name]_[id_2]_[expname]_[epoch]_G[id_2].csv \
                                                        --stats stats/[name]_[id_2]_[expname]_[epoch]_G[id_2]_S[id_1]_stats
```

- Run the python command for the train set and test set
- Can add ```CUDA_VISIBLE_DEVICES="0,1"``` before python if possible (embedding takes a while)

#### 3.3. Compute Frechet Distance between stats of the real and generated.

```
python -m frechet_audio_distance.compute_fad --background_stats stats/[name]_[id_2]_stats \
                                             --test_stats stats/[name]_[id_2]_[expname]_[epoch]_G[id_2]_S[id_1]_stats
```

- Background refers to the real training audio set.
- Test refers to the fake style transferred audio set.

---

## Pretrained Models

With respect to the current data preperation set up, the following models were trained:

|  Model  |                                      Flickr                                      |                                       URMP                                       |
|:-------:|:--------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------:|
| VAE-GAN | [link](https://drive.google.com/drive/folders/1Wui2Pt4sOBl71exRh49GX_JEBpFv_vNg) | [link](https://drive.google.com/drive/folders/1Nq3tKE-kcoMOw5AYEa0qWddxwbUYL8aA) |
| WaveNet | [link](https://drive.google.com/drive/folders/1uK_mAfA24sXkru7hmBLJCtqRQIq43zNx) | [link](https://drive.google.com/drive/folders/1RZhP6UFiJjZBYL4JyIf4tOTa1RrX-l0a) |

#### Pretrained VAE-GAN

1. Create directory ```voice_conversion/src/saved_models/initial```.
2. Drag .pth files to that directory.
3. Call with ```--model_name initial --epoch 99``` for inference (with epoch 490 for URMP).

Notes
- G1 is female for Flickr, trumpet for URMP.
- G2 is male for Flickr, violin for URMP.

#### Pretrained WaveNet

1. Create directory ```wavenet_vocoder/egs/gaussian/exp/```
2. Drag the folder such as ```flickr_1_train_no_dev_flickr``` into that directory.
3. Call ./infer.sh with appropriate arguments such as ```spk="flickr_1" inferdir="initial_99_G1_S2"```.

Notes
- flickr_1 is female, flickr_2 is male.
- urmp_1 is trumpet, urmp_2 is violin.
- inferdir is just a local directory in ```voice_conversion/src/out_infer/```. You can point it to any local dir within that path for input.
