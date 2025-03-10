'''
Train the model with PyTorch Lightning
Compatible GPUs that support bf16 training (e.g., RTX3090)

Usage: python lightning_train.py /path/to/hparam/file.yaml

Author: Longshen Ou, 2024/07/25
'''

import sys
import torch
import mlconfig
import lightning as L
from utils import jpath
from solo_datasets import get_dataloader
from lightning_model import get_lit_model
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

if __name__ == '__main__':
    seed_everything(42)
    
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True


def main():
    if not len(sys.argv) == 2: # Debug
        config_fp = '/home/longshen/work/SoloDetection/hparams/cross_dataset/train_on_mjn_inst.yaml'
        config = mlconfig.load(config_fp)
        config['train']['fast_dev_run'] = 6
        config.dataset['num_workers'] = 0
        config['debug'] = True
    else:
        config_fp = sys.argv[1]
        config = mlconfig.load(config_fp)

    # Setup data
    train_loader = get_dataloader(config, 'train')
    valid_loader = get_dataloader(config, 'valid')

    # Instantiate the model
    config.model['n_inst_type'] = train_loader.dataset.inst_tokenizer.vocab_size() # Adjust model's vocab size to same as dataset
    config.model['na_inst_id'] = train_loader.dataset.inst_tokenizer.inst2id['na'] # Update na's id to the same as dataset
    lit_model = get_lit_model(config)

    # Train the model
    out_dir = jpath(config.result_root, config.out_dir)
    checkpoint_callback = ModelCheckpoint(
        monitor='valid_f1',
        mode="max",
        filename='{epoch:02d}-{valid_f1:.4f}-{valid_loss:.4f}',
        save_top_k=1,
    )
    earlystop_callback = EarlyStopping(
        monitor='valid_f1',
        mode='max',
        patience=config.train['earlystop_patience'] if 'earlystop_patience' in config.train else 10,
    )
    trainer = L.Trainer(
        max_epochs=config.train.n_epoch,
        default_root_dir=out_dir, # output and log dir
        callbacks=[checkpoint_callback, earlystop_callback],
        fast_dev_run=config.train['fast_dev_run'] if 'fast_dev_run' in config.train else False,
        accelerator='gpu',
        precision='bf16', 
        val_check_interval=config.train['val_check_interval'] if 'val_check_interval' in config.train else 0.2,
        accumulate_grad_batches=config.train['grad_accumulate'] if 'grad_accumulate' in config.train else 1,
    )
    trainer.fit(
        model=lit_model,
        train_dataloaders=train_loader, 
        val_dataloaders=valid_loader
    )
    
    # Do thet testing
    if config.get('debug', False) is False:
        test_loader = get_dataloader(config, 'test')
        trainer.test(model=lit_model, dataloaders=valid_loader, ckpt_path='best')
        trainer.test(model=lit_model, dataloaders=test_loader, ckpt_path='best')





if __name__ == '__main__':
    main()