'''
Test the model's performance with PyTorch Lightning
Using the test loop defined in the lightning model

Usage: python lightning_test.py /path/to/hparam/file.yaml

Author: Longshen Ou, 2024/07/25
'''

import sys
import torch
import mlconfig
import lightning as L
from solo_datasets import get_dataloader
from lightning_model import load_lit_model
from lightning.pytorch import seed_everything

torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True


def main():
    seed_everything(42, workers=True)

    if not len(sys.argv) == 2: # Debug
        config_fp = '/home/longshen/work/SoloDetection/hparams/cross_dataset/train_on_medley.yaml'
        config = mlconfig.load(config_fp)
        config['train']['fast_dev_run'] = 6
        config.dataset['num_workers'] = 0
        config['debug'] = True
    else:
        config_fp = sys.argv[1]
        config = mlconfig.load(config_fp)

    # Load a lightning model from checkpoint
    lit_model = load_lit_model(config)

    # Setup data
    valid_loader = get_dataloader(config, 'valid')
    test_loader = get_dataloader(config, 'test')

    # Prepare trainer for testing
    trainer = L.Trainer(
        logger=False,
        fast_dev_run=config['fast_dev_run'] if 'fast_dev_run' in config else False,
        # precision='bf16',
        accelerator="gpu",
        devices=1,
    )
    
    print('Validation set performance:')
    trainer.test(
        model=lit_model,
        dataloaders=valid_loader,
    )
    print('Test set performance:')
    trainer.test(
        model=lit_model,
        dataloaders=test_loader,
    )


if __name__ == '__main__':
    main()