'''
Test the dataset by iterating over it.

Author: Longshen Ou, 2024/07/25
'''

import os
import sys

sys.path.append('.')
sys.path.append('..')

import torch
import mlconfig
import torch.nn as nn
from solo_datasets import get_dataloader
from solo_models import get_model
from tqdm import tqdm


def main():
    config_fp = '/home/longshen/work/SoloDetection/hparams/cross_dataset/train_on_medley.yaml'
    config = mlconfig.load(config_fp)

    model = get_model(config.model)
    dataloader = get_dataloader(config, 'train', num_workers=0)

    for batch in tqdm(dataloader):
        audios = batch['audio']
        inst_ids = batch['inst_ids']
        labels = batch['label']

        # # Test model
        # out = model(audios, insts=inst_ids) # [B, n_frame, n_inst_type=15]
        # out = out.permute(0, 2, 1)
        # loss = nn.functional.cross_entropy(out, labels)

        a = 2

if __name__ == '__main__':
    main()