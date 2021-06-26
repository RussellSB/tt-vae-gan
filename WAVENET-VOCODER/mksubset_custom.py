# coding: utf-8
"""
Make subset of dataset, but split so that subsamples can be concattenated to full samples later

usage: mksubset.py [options] <in_dir> <out_dir>

options:
    -h, --help               Show help message.
    --limit=<N>              Limit dataset size by N-hours [default: 10000].
    --dev-size=<N>           Development size or rate [default: 0.1].
    --test-size=<N>          Test size or rate [default: 0.1].
"""
from docopt import docopt
from shutil import copy
from os.path import join
import glob
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm

if __name__ == "__main__":
    args = docopt(__doc__)
    in_dir = args['<in_dir>']
    out_dir = args['<out_dir>']
    limit = float(args['--limit'])
    dev_size = float(args['--dev-size'])
    test_size = float(args['--test-size'])
    
    # Making directory and subdirectory if they doesn't exist
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(join(out_dir, 'train_no_dev'), exist_ok=True)
    os.makedirs(join(out_dir, 'dev'), exist_ok=True)
    os.makedirs(join(out_dir, 'eval'), exist_ok=True)

    # Firstly get matchable identities across parts (assumes all exp ids have same ids, bases it on first)
    ids = []
    for f in os.listdir(in_dir):
        identity = '_'.join(f.split('_')[:-1])
        if identity not in ids: ids.append(identity)

    # Performing the split based on unique identity        
    print('Splitting based on ' + str(len(ids)) + ' full audio samples...') 
    train_ids, dev_test_ids = train_test_split(ids, test_size=test_size + dev_size)
    dev_ids, test_ids = train_test_split(dev_test_ids, test_size = test_size / (test_size + dev_size))
    print('Train samples:', len(train_ids))
    print('Dev samples:', len(dev_ids))
    print('Test samples:', len(test_ids))
    print()

    # Making an iterable with destination path
    id_sets = [(train_ids, join(out_dir, 'train_no_dev/')),
               (dev_ids, join(out_dir, 'dev/')),
               (test_ids, join(out_dir, 'eval/'))]

    # Copying files
    print('Copying files of train, dev, and test set respectively...')
    for id_set, dst_path in id_sets:
        print('Copying to', dst_path)
        for i in tqdm(id_set):
            path_subsamples = sorted(glob.glob(join(in_dir,i)+'*.wav'))
            for src_file in path_subsamples:
                copy(src_file, dst_path)

    print('Finished!')