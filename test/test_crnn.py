'''
Test the CRNN model with dummy inputs.

Author: Longshen Ou, 2024/07/25
'''

import os
import sys

sys.path.append('.')
sys.path.append('..')

from solo_models import *
import torch
import torchaudio
import mlconfig


def main():
    # model = MultiChannelCRNN()

    config_fp = '/home/longshen/work/SoloDetection/hparams/baselines/crnn_mert.yaml'
    config = mlconfig.load(config_fp)

    model = MultiChannelMertCRNN(config.model)

    dummy_input = torch.randn(4, 10, 120320) # [bs, n_ch, n_sample]
    out = model(dummy_input)

    b = 2


def procedures():
    pass


if __name__ == '__main__':
    main()