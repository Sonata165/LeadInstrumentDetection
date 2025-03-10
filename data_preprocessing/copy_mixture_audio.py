'''
Copy mixture audio to a separate dir
To facilitate msaf's processing of segmentation features
For the SVM baseline

Author: Longshen Ou, 2024/07/25
'''

import os
import sys

sys.path.append('..')

import shutil
from utils import *
from tqdm import tqdm

def main():
    stem_audio_dir = '/Users/longshen/Code/Datasets/MJN/Preliminary/stem_audio'
    save_dir = '/Users/longshen/Code/Datasets/MJN/Preliminary/mix_audio'
    pfm_ids = ls(stem_audio_dir)
    for pfm_id in tqdm(pfm_ids):
        pfm_dir = jpath(stem_audio_dir, pfm_id)
        audio_fns = ls(pfm_dir)
        mix_fn = [fn for fn in audio_fns if fn.startswith('m') or fn.startswith('M')]
        assert len(mix_fn) == 1

        mix_fp = jpath(pfm_dir, mix_fn[0])

        save_fp = jpath(save_dir, pfm_id + '.wav')
        shutil.copy(mix_fp, save_fp)


if __name__ == '__main__':
    main()
