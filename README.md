# Timbre Transfer with VAE-GAN & WaveNet

This pipeline follows and extends the work of [Albadawy & Lyu 2020](https://ebadawy.github.io/post/speech_style_transfer/Albadawy_et_al-2020-INTERSPEECH.pdf). The work that used this shows (amongst other things) that their proposed **voice** conversion model is also applicable to context of musical instruments, therefore reforming the conversion to a more generalised audio style - **timbre**. 

You can find the pre-print of this work [here](https://arxiv.org/abs/2109.02096). Please be sure to reference it if you use this code for your research:

```
@misc{sammutbonnici2021timbre,
      title={Timbre Transfer with Variational Auto Encoding and Cycle-Consistent Adversarial Networks}, 
      author={Russell Sammut Bonnici and Charalampos Saitis and Martin Benning},
      year={2021},
      eprint={2109.02096},
      archivePrefix={arXiv},
      primaryClass={cs.SD}
}
```

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

**Female to Male:**

- [Input](https://user-images.githubusercontent.com/35470600/131001275-b5dcd806-a2cf-451b-a61f-56ab453af66f.mp4) -> [Griffin Lim Output](https://user-images.githubusercontent.com/35470600/131001312-238a13d4-0d6b-451a-afbf-284979414daf.mp4) -> [WaveNet Output](https://user-images.githubusercontent.com/35470600/131001361-f285ae93-d1f6-4325-84c1-b3982dcc14dc.mp4)

![G2_970641406_9a20ee636a_4](https://user-images.githubusercontent.com/35470600/131000264-2527a5dc-95b2-408a-beeb-8fc662db6590.png)

**Violin to Trumpet:**

- [Input](https://drive.google.com/file/d/1idIHs7MOv2NpMy6agDlc9alWTlRq2f-c/view?usp=sharing) -> [Griffin Lim Output](https://drive.google.com/file/d/1r4ArPQuwtwMJA41aCji_iy_3kvOkGcCy/view?usp=sharing) -> [WaveNet Output](https://drive.google.com/file/d/13u8bUUbIGG7Z8hupQNbYX1VYRH1HmTp1/view?usp=sharing)

![G1_AuSep_2_vn_32_Fugue](https://user-images.githubusercontent.com/35470600/131002898-d1234b06-9aa6-4adc-9268-6de0200cc9bf.png)

## Hardware
Recommended GPU VRAM per model:
- voice_conversion - **2 GB**
- wavenet_vocoder - **8 GB**
- fad - **24 GB** 

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

- For two speakers ids this would be either ```1``` or ```2```. If you want to train all, make [id] as "_all" or something
- You need to run the .sh command for each target timbre if you want to train specific vocoders.
- Note that alternate to specific vocoders train general vocoders (a vocoder for all speakers instead of for each speaker). For this prepare training and test data in the previous step from all speakers in one dataset.

#### 2.3. Train a wavenet vocoder.

```
spk="[name]_[id]" hparams=conf/[name].json ./run.sh --stage 2 --stop-stage 2 
```

- Just like preprocessing, you need to run this for each target timbre for specific vocoding.
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

With respect to the current data preperation set up, the following one-to-one VAE-GANs and specific vocoders were trained:

|  Model  |                                      Flickr                                      |                                       URMP                                       |
|:-------:|:--------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------:|
| VAE-GAN | [link](https://drive.google.com/drive/folders/1Wui2Pt4sOBl71exRh49GX_JEBpFv_vNg) | [link](https://drive.google.com/drive/folders/1Nq3tKE-kcoMOw5AYEa0qWddxwbUYL8aA) |
| WaveNet | [link](https://drive.google.com/drive/folders/1uK_mAfA24sXkru7hmBLJCtqRQIq43zNx) | [link](https://drive.google.com/drive/folders/1RZhP6UFiJjZBYL4JyIf4tOTa1RrX-l0a) |

#### Pretrained VAE-GAN

1. Create directory ```voice_conversion/src/saved_models/initial```.
2. Drag .pth files to that directory.
4. Call with ```--model_name initial --epoch 99``` for inference (with epoch 490 for URMP).

Notes
- G1 is female for Flickr, trumpet for URMP.
- G2 is male for Flickr, violin for URMP.

#### Pretrained WaveNet

1. Create directory ```wavenet_vocoder/egs/gaussian/exp/```
2. Drag the folder such as ```flickr_1_train_no_dev_flickr``` into that directory.
3. Drag the meanvar.joblib file within the folder to a new directory following ```wavenet_vocoder/egs/gaussian/dump/[spk]/logmelspectrogram/org``` - where ```[spk]``` corresponds to ```flickr_1``` for example.
4. Call ./infer.sh with appropriate arguments such as ```spk="flickr_1" inferdir="initial_99_G1_S2"```.

Notes
- flickr_1 is female, flickr_2 is male.
- urmp_1 is trumpet, urmp_2 is violin.
- inferdir is just a local directory in ```voice_conversion/src/out_infer/```. You can point it to any local dir within that path for input.
