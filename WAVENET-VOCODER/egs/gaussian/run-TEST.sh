#!/bin/bash

script_dir=$(cd $(dirname ${BASH_SOURCE:-$0}); pwd)
VOC_DIR=$script_dir/../../

# Directory that contains all wav files
# **CHANGE** this to your database path
db_root='../../../pool/wavs-full-test-4-TEST/'
datadir=lj # jj, tt, tt-2
spk="test-4" # 'test-7, test-4, test-vn, test-tpt
outdir=dump/$datadir/logmelspectrogram/norm/$spk
dumpdir=dump

# THIS IS PURELY FOR PREPROCESSING FULL SAMPLE TEST SETS (for full demo after)

# train/dev/eval split
dev_size=0.1
eval_size=0.1
# Maximum size of train/dev/eval data (in hours).
# set small value (e.g. 0.2) for testing
limit=1000000

# waveform global gain normalization scale
global_gain_scale=0.55

stage=0
stop_stage=0

# Hyper parameters (.json)
# **CHANGE** here to your own hparams
hparams=conf/gaussian_wavenet.json

# Batch size at inference time.
inference_batch_size=32
# Leave empty to use latest checkpoint
eval_checkpoint=
# Max number of utts. for evaluation( for debugging)
eval_max_num_utt=1000000

# exp tag
tag="main" # tag for managing experiments.

. $VOC_DIR/utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train_no_dev"
dev_set="dev"
eval_set="eval"
datasets=($train_set $dev_set $eval_set)

# exp name
if [ -z ${tag} ]; then
    expname=${spk}_${train_set}_$(basename ${hparams%.*})
else
    expname=${spk}_${train_set}_${tag}
fi
expdir=exp/$expname

feat_typ="logmelspectrogram"

# Output directories
data_root=data/$spk                        # train/dev/eval splitted data
dump_org_dir=$dumpdir/$spk/$feat_typ/org   # extracted features (pair of <wave, feats>)
dump_norm_dir=$dumpdir/$spk/$feat_typ/norm # extracted features (pair of <wave, feats>)

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: train/dev/eval split"
    if [ -z $db_root ]; then
      echo "ERROR: DB ROOT must be specified for train/dev/eval splitting."
      echo "  Use option --db-root \${path_contains_wav_files}"
      exit 1
    fi
    python $VOC_DIR/mksubset.py $db_root $data_root \
      --train-dev-test-split --dev-size $dev_size --test-size $eval_size \
      --limit=$limit
      
    # ADDED
    # Copies wavs from all other folders to training folder
    echo Copying files from dev and eval to train_no_dev
    cp -a data/$spk/dev/. data/$spk/train_no_dev/
    cp -a data/$spk/eval/. data/$spk/train_no_dev/
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: Feature Generation"
    python $VOC_DIR/preprocess.py wavallin $data_root/$train_set ${dump_org_dir}/$train_set \
        --hparams="global_gain_scale=${global_gain_scale}" --preset=$hparams

    # Compute mean-var normalization stats
    find $dump_org_dir/$train_set -type f -name "*feats.npy" > train_list.txt
    python $VOC_DIR/compute-meanvar-stats.py train_list.txt $dump_org_dir/meanvar.joblib
    rm -f train_list.txt

    # Apply normalization
    python $VOC_DIR/preprocess_normalize.py ${dump_org_dir}/$train_set $dump_norm_dir/$train_set \
        $dump_org_dir/meanvar.joblib
    cp -f $dump_org_dir/meanvar.joblib ${dump_norm_dir}/meanvar.joblib
    
    echo Saving to specified $outdir
    mkdir -p $outdir
    cp -r $dump_norm_dir/$train_set/. $outdir
fi
