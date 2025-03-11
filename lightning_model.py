'''
Define the lightning model for training with PyTorch Lightning

Author: Longshen Ou, 2024/07/25
'''

import torch
import transformers
import torch.nn as nn
import lightning as L
from torch import optim
from utils import jpath, get_latest_checkpoint
from solo_models import get_model
from sklearn.metrics import f1_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import LinearLR


def load_lit_model(config):
    '''
    Load a lightning model from the latest Lightning checkpoint
    '''
    out_dir = jpath(config['result_root'], config['out_dir'])
    if 'ckpt_fp' not in config['model'] or config['model']['ckpt_fp'] is None:
        latest_version_dir, ckpt_fp = get_latest_checkpoint(out_dir)
    else:
        ckpt_fp = config['model']['ckpt_fp']
    lit_model_cls = eval(config.model.lit_model) if 'lit_model' in config.model else LitMertSoloInst
    l_model = lit_model_cls.load_from_checkpoint(ckpt_fp, model_config=config.model, train_config=config.train, infer=True)
    return l_model


def get_lit_model(config):
    '''
    Create a lightning model
    '''
    lit_model_cls = eval(config.model.lit_model) if 'lit_model' in config.model else LitMertSoloInst
    lit_model = lit_model_cls(model_config=config.model, train_config=config.train)
    return lit_model


class LitMertSoloInst(L.LightningModule):
    def __init__(self, model_config, train_config, infer=False):
        super().__init__()
        self.model = get_model(model_config)
        
        # Freeze parameters I want
        if model_config['freeze_mert'] is True:
            for param in self.model.mert.parameters():
                param.requires_grad = False

            if model_config.get('train_last_layer', False):
                for param in self.model.mert.encoder.layers[-1].parameters():
                    param.requires_grad = True

        self.model_config = model_config
        self.train_config = train_config

        self.valid_out = []
        self.valid_tgt = []

        self.na_inst_id = model_config.get('na_inst_id', 11) # Default to 11 (MJN's na inst id value)

        self.save_hyperparameters()


    def training_step(self, batch, batch_idx):
        
        audios = batch['audio']
        inst_ids = batch['inst_ids']
        labels = batch['label']

        out = self.model(audios, insts=inst_ids) # [B, n_frame, n_inst_type=15]

        # Loss computation
        out = out.permute(0, 2, 1)
        loss = nn.functional.cross_entropy(out, labels)
        
        # Logging to TensorBoard (if installed) by default
        scheduler = self.lr_schedulers()
        self.log("train_loss", loss)
        self.log('train_lr', scheduler.get_last_lr()[0])

        # For linear scheduler
        if isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR):
            scheduler.step()

        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.evaluation_step(batch, split='valid')
        return loss
    
    def test_step(self, batch, batch_idx):
        loss = self.evaluation_step(batch, split='test')
        return loss
    
    def evaluation_step(self, batch, split):
        '''
        Evaluation loop shared by validation step and test step
        '''
        audios = batch['audio']
        inst_ids = batch['inst_ids']
        labels = batch['label'] # [B, n_frame]
        out = self.model(audios, insts=inst_ids) # [B, n_frame, n_inst_type=15]

        # Loss computation
        out = out.permute(0, 2, 1)  # [B, n_inst_type, n_frame]
        loss = nn.functional.cross_entropy(out, labels)

        # Accuracy computation
        preds = out.argmax(dim=1)
        acc = (preds == labels).float().mean()

        # Prepare for macro F1 score computation
        self.valid_out.extend(preds.tolist())
        self.valid_tgt.extend(labels.tolist())

        # Compute sample-level F1 score, average within batch
        f1s = []
        for label, pred in zip(labels, preds):
            f1 = f1_score(label.cpu(), pred.cpu(), average='macro')
            f1s.append(f1)
        batch_f1 = sum(f1s) / len(f1s)

        # Logging to TensorBoard (if installed) by default
        bs = audios.shape[0]
        self.log("{}_f1".format(split), batch_f1)
        self.log("{}_loss".format(split), loss, batch_size=bs)
        self.log("{}_acc".format(split), acc, batch_size=bs)

        return loss
    
    def test_step(self, batch, batch_idx):
        self.evaluation_step(batch, split='test')
    
    def on_validation_epoch_end(self):
        scheduler = self.lr_schedulers()

        # If the selected scheduler is a ReduceLROnPlateau scheduler.
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(self.trainer.callback_metrics["valid_f1"])

    def on_train_epoch_end(self):
        scheduler = self.lr_schedulers()
        if isinstance(scheduler, torch.optim.lr_scheduler.ExponentialLR):
            scheduler.step()
            a = 2

    def configure_optimizers(self):
        # optimizer = optim.AdamW(
        #     self.parameters(), 
        #     # self.model.inst_cls_head.parameters(),
        #     lr=float(self.train_config.lr),
        #     weight_decay=self.train_config['weight_decay']
        # )

        # Different learning rate for different parts of the model
        if self.model_config.model_class == 'MertSoloDetectorInstEmb':
            optimizer = optim.AdamW([
                {'params': self.model.inst_cls_head.parameters(), 'lr': float(self.train_config.lr), 'weight_decay':self.train_config['weight_decay']},
                {'params': self.model.inst_emb.parameters(), 'lr': float(self.train_config.lr)},
                {'params': self.model.mert.parameters(), 'lr': float(self.train_config.mert_lr), 'weight_decay':self.train_config['weight_decay']},
            ])
        elif self.model_config.model_class == 'MertSoloDetectorByChannel':
            optimizer = optim.AdamW([
                {'params': self.model.inst_cls_head.parameters(), 'lr': float(self.train_config.lr), 'weight_decay':self.train_config['weight_decay']},
                {'params': self.model.channel_attn.parameters(), 'lr': float(self.train_config.attn_lr), 'weight_decay':self.train_config['weight_decay']},
                {'params': self.model.inst_emb.parameters(), 'lr': float(self.train_config.lr)},
                {'params': self.model.mert.parameters(), 'lr': float(self.train_config.mert_lr), 'weight_decay':self.train_config['weight_decay']},
            ])
        elif self.model_config.model_class in ['MertSoloDetectorFromMix', 'MertSoloDetectorFromMixSegment']:
            optimizer = optim.AdamW([
                {'params': self.model.inst_cls_head.parameters(), 'lr': float(self.train_config.lr), 'weight_decay':self.train_config['weight_decay']},
                {'params': self.model.mert.parameters(), 'lr': float(self.train_config.mert_lr), 'weight_decay':self.train_config['weight_decay']},
            ])
        elif self.model_config.model_class in ['MertSoloDetectorChattnChcls', 'MertSoloDetectorChattnChclsNoMix']:
            optimizer = optim.AdamW([
                {'params': self.model.ch_cls_head.parameters(), 'lr': float(self.train_config.lr), 'weight_decay':self.train_config['weight_decay']},
                {'params': self.model.channel_attn.parameters(), 'lr': float(self.train_config.attn_lr), 'weight_decay':self.train_config['weight_decay']},
                {'params': self.model.inst_emb.parameters(), 'lr': float(self.train_config.lr)},
                {'params': self.model.ch_emb.parameters(), 'lr': float(self.train_config.lr)},
                {'params': self.model.mert.parameters(), 'lr': float(self.train_config.mert_lr), 'weight_decay':self.train_config['weight_decay']},
            ])
        elif self.model_config.model_class == 'MertSoloDetectorChattnChclsRnn':
            optimizer = optim.AdamW([
                {'params': self.model.ch_cls_head.parameters(), 'lr': float(self.train_config.lr), 'weight_decay':self.train_config['weight_decay']},
                {'params': self.model.channel_attn.parameters(), 'lr': float(self.train_config.attn_lr), 'weight_decay':self.train_config['weight_decay']},
                {'params': self.model.inst_emb.parameters(), 'lr': float(self.train_config.lr)},
                {'params': self.model.ch_emb.parameters(), 'lr': float(self.train_config.lr)},
                {'params': self.model.mert.parameters(), 'lr': float(self.train_config.mert_lr), 'weight_decay':self.train_config['weight_decay']},
                {'params': self.model.rnn.parameters(), 'lr': float(self.train_config.rnn_lr), 'weight_decay':self.train_config['weight_decay']},
            ])
        elif self.model_config.model_class in ['MultiChannelCRNN', 'MultiChannelMertCRNN', 'MultiChannelMertAttnCRNN']:
            optimizer = optim.AdamW(self.model.parameters(), lr=float(self.train_config.lr), weight_decay=self.train_config['weight_decay'])
        
        scheduler_type = self.train_config.get('lr_scheduler', 'anneal')

        if scheduler_type == 'exponential':
            # ExponentialLR
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=self.train_config.gamma
            )
            ret = {"optimizer": optimizer, "lr_scheduler": scheduler},
        elif scheduler_type == 'linear':
            # Linear scheduler
            max_steps = self.num_training_steps()
            scheduler = transformers.get_linear_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=self.train_config['warmup_steps'],
                num_training_steps=max_steps,
            )
            # For linear scheduler
            ret = {"optimizer": optimizer, "lr_scheduler": scheduler},
        elif scheduler_type == 'anneal':
            # Annealing
            anneal_scheduler = ReduceLROnPlateauPatch(
                optimizer,
                mode='max',
                factor=0.5,
                patience=self.train_config['lr_anneal_patience'] if 'lr_anneal_patience' in self.train_config else 3,
                verbose=True
            )
            # For ReduceLROnPlateau
            ret = {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": anneal_scheduler,
                    "monitor": "valid_f1",
                },
            }
        else:
            # For no scheduler
            ret = {"optimizer": optimizer}

        return ret
    
    def num_training_steps(self) -> int:
        self.trainer.reset_train_dataloader(self)

        """Get number of training steps"""
        if self.trainer.max_steps > -1:
            return self.trainer.max_steps

        # self.trainer.fit_loop.setup_data()
        dataset_size = len(self.trainer.train_dataloader)
        num_steps = dataset_size * self.trainer.max_epochs

        return num_steps

    def get_step_per_epoch(self):
        if self.trainer.train_dataloader is not None:
            return len(self.trainer.train_dataloader)
        self.trainer.fit_loop.setup_data()
        return len(self.trainer.train_dataloader)
    

class LitMertSoloChannel(LitMertSoloInst):
    
    def validation_step(self, batch, batch_idx):
        audios = batch['audio'] # [B, n_ch, n_samples=120320]
        inst_ids = batch['inst_ids'] # [B, n_ch]
        labels = batch['label'] # [B, n_frame]
        out = self.model(audios, insts=inst_ids) # [B, n_frame, n_inst_type=15]

        # Loss computation
        out = out.permute(0, 2, 1)  # [B, n_inst_type, n_frame]
        loss = nn.functional.cross_entropy(out, labels)

        # Accuracy computation
        # mask out channel > len(audio) + 1   (na + all track (including mix))
        out[:, audios.size(1) + 1:] = -1e6
        preds = out.argmax(dim=1) # [B, n_frame]
        acc = (preds == labels).float().mean()

        # Prepare for macro F1 score computation
        self.valid_out.extend(preds.tolist())
        self.valid_tgt.extend(labels.tolist())

        # Compute sample-level F1 score, average within batch
        f1s = []
        for label, pred in zip(labels, preds):
            f1 = f1_score(label.cpu(), pred.cpu(), average='macro')
            f1s.append(f1)
        batch_f1 = sum(f1s) / len(f1s)
        self.log("valid_f1", batch_f1)

        # Compute inst F1 score
        inst_f1s = []
        na_inst_id = self.na_inst_id
        for label, pred, ch_to_inst in zip(labels, preds, inst_ids): # label: [n_frame], inst_map: [n_ch]
            inst_id_tgt = [ch_to_inst[i-1].item() if i > 0 else na_inst_id for i in label]
            inst_id_out = [ch_to_inst[i-1].item() if i > 0 else na_inst_id for i in pred]
            inst_f1 = f1_score(inst_id_out, inst_id_tgt, average='macro')
            inst_f1s.append(inst_f1)
        inst_f1 = sum(inst_f1s) / len(inst_f1s)
        self.log("valid_inst_f1", inst_f1)

        # Logging to TensorBoard (if installed) by default
        bs = audios.shape[0]
        self.log("valid_loss", loss, batch_size=bs)
        self.log("valid_acc", acc, batch_size=bs)
        # self.log("valid_f1", f1, batch_size=bs)

        return loss
    
    def test_step(self, batch, batch_idx):
        audios = batch['audio'] # [B, n_ch, n_samples=120320]
        inst_ids = batch['inst_ids'] # [B, n_ch]
        labels = batch['label'] # [B, n_frame]
        out = self.model(audios, insts=inst_ids) # [B, n_frame, n_inst_type=15]

        # Loss computation
        out = out.permute(0, 2, 1)  # [B, n_inst_type, n_frame]
        loss = nn.functional.cross_entropy(out, labels)

        # Accuracy computation
        # mask out channel > len(audio) + 1
        out[:, audios.size(1) + 1:] = -1e6
        preds = out.argmax(dim=1) # [B, n_frame]
        acc = (preds == labels).float().mean()

        # Prepare for macro F1 score computation
        self.valid_out.extend(preds.tolist())
        self.valid_tgt.extend(labels.tolist())

        # Compute sample-level F1 score, average within batch
        f1s = []
        for label, pred in zip(labels, preds):
            f1 = f1_score(label.cpu(), pred.cpu(), average='macro')
            f1s.append(f1)
        batch_f1 = sum(f1s) / len(f1s)
        self.log("1_test_f1", batch_f1)

        # Compute inst F1 score
        inst_f1s = []
        na_inst_id = self.na_inst_id
        for label, pred, ch_to_inst in zip(labels, preds, inst_ids): # label: [n_frame], inst_map: [n_ch]
            inst_id_tgt = [ch_to_inst[i-1].item() if i > 0 else na_inst_id for i in label]
            inst_id_out = [ch_to_inst[i-1].item() if i > 0 else na_inst_id for i in pred]
            inst_f1 = f1_score(inst_id_out, inst_id_tgt, average='macro')
            inst_f1s.append(inst_f1)
        inst_f1 = sum(inst_f1s) / len(inst_f1s)
        self.log("2_test_inst_f1", inst_f1)

        # Logging to TensorBoard (if installed) by default
        bs = audios.shape[0]
        self.log("0_test_loss", loss, batch_size=bs)
        self.log("3_test_acc", acc, batch_size=bs)
        

class LitMertSoloGuitar(LitMertSoloInst):
    def training_step(self, batch, batch_idx):
        
        audios = batch['audio']
        inst_ids = batch['inst_ids']
        labels = batch['label']

        out = self.model(audios, insts=inst_ids) # [B, n_frame, n_inst_type=15]

        # Loss computation
        loss = nn.functional.cross_entropy(out, labels)
        
        # Logging to TensorBoard (if installed) by default
        scheduler = self.lr_schedulers()
        self.log("train_loss", loss)
        self.log('train_lr', scheduler.get_last_lr()[0])

        # For linear scheduler
        if isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR):
            scheduler.step()

        return loss

    def evaluation_step(self, batch, split):
        '''
        Evaluation loop shared by validation step and test step
        '''
        audios = batch['audio']
        inst_ids = batch['inst_ids']
        labels = batch['label'] # [B, n_frame]
        out = self.model(audios, insts=inst_ids) # [B, n_frame, n_inst_type=15]

        # Loss computation
        loss = nn.functional.cross_entropy(out, labels)

        # Accuracy computation
        preds = out.argmax(dim=1)
        acc = (preds == labels).float().mean()

        # Prepare for macro F1 score computation
        self.outs.extend(preds.tolist())
        self.tgts.extend(labels.tolist())

        # Logging to TensorBoard (if installed) by default
        bs = audios.shape[0]
        self.log("{}_loss".format(split), loss, batch_size=bs)
        self.log("{}_acc".format(split), acc, batch_size=bs)

        return loss
    
    def on_validation_epoch_start(self) -> None:
        self.outs = []
        self.tgts = []


    def on_test_epoch_start(self) -> None:
        self.outs = []
        self.tgts = []

    def on_validation_epoch_end(self):
        f1 = f1_score(self.tgts, self.outs)
        macro_f1 = f1_score(self.tgts, self.outs, average='macro')

        self.log("valid_f1", f1)
        self.log("valid_macro_f1", macro_f1)
        
        scheduler = self.lr_schedulers()
        # If the selected scheduler is a ReduceLROnPlateau scheduler.
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(self.trainer.callback_metrics["valid_f1"])

    def on_test_epoch_end(self) -> None:
        f1 = f1_score(self.tgts, self.outs)
        macro_f1 = f1_score(self.tgts, self.outs, average='macro')

        self.log("test_f1", f1)
        self.log("test_macro_f1", macro_f1)
        
        scheduler = self.lr_schedulers()
        # If the selected scheduler is a ReduceLROnPlateau scheduler.
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(self.trainer.callback_metrics["valid_f1"])


class ReduceLROnPlateauPatch(ReduceLROnPlateau, _LRScheduler):
    def get_last_lr(self):
        return [ group['lr'] for group in self.optimizer.param_groups ]