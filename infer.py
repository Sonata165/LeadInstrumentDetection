'''
The inference script for the frame-level solo detection task
With either instrument classification or channel classification model

Usage: python infer.py /path/to/hparam/file.yaml
Author: Longshen Ou, 2024/07/25
'''
import sys
import torch
import mlconfig 
from tqdm import tqdm
from solo_datasets import *
from lightning_model import *
from lightning_train import get_dataloader
from lightning.pytorch import seed_everything
from utils import jpath, get_latest_checkpoint, save_json

if __name__ == '__main__':
    seed_everything(42)

torch.backends.cuda.matmul.allow_tf32 = True


def main():
    if not len(sys.argv) == 2:
        config_fp = '/home/longshen/work/SoloDetection/hparams/mjn_chcls/ch_delete.yaml'
        config = mlconfig.load(config_fp)
        config['train']['fast_dev_run'] = 5
        debug = True
    else:
        config_fp = sys.argv[1]
        config = mlconfig.load(config_fp)
        debug = False

    chcls = True
    split = 'valid' # 'valid' or 'test'

    # Load a lightning model from checkpoint
    lit_model = load_lit_model(config)
    model = lit_model.model
    model.eval() # class: MuseCocoLMHeadModel
    model.cuda()

    # Prepare the test set
    
    test_loader = get_dataloader(config, split=split)
    metric = Metric()

    # Iterate over test set, do the classification, save the output
    outs = []
    tgts = []
    cnt = 0
    with torch.no_grad():
        pbar = tqdm(test_loader)
        for id, batch in enumerate(pbar):
            pbar.set_description(str(id))

            audios = batch['audio'].cuda()
            inst_ids = batch['inst_ids'].cuda()
            labels = batch['label']
            out = model(audios, insts=inst_ids).cpu()  # [B, n_frame, n_inst_type=15]

            # Mask out logits for non-existing channels
            if chcls is True:
                n_channels = audios.size(1)
                out[:, :, n_channels + 1:] = -1e6

            preds = out.argmax(dim=2) # [B, n_frame]

            # Convert the channel output and label to instrument id
            if chcls is True:
                na_inst_id = 11
                new_out, new_labels = [], []
                for out_i, label_i, inst_id in zip(preds, labels, inst_ids):
                    new_out_i = [inst_id[j-1].item() if j != 0 else na_inst_id for j in out_i]
                    new_out_j = [inst_id[j-1].item() if j != 0 else na_inst_id for j in label_i]
                    new_out.append(new_out_i)
                    new_labels.append(new_out_j)
                preds = torch.tensor(new_out).cuda()
                labels = torch.tensor(new_labels).cuda()

            # Accuracy computation
            acc = (preds == labels).float().mean()
            metric.update('acc', acc.item())

            # Save the output and target
            outs.extend(preds.tolist())
            tgts.extend(labels.tolist())

            # Sample-level Macro F1 score computation
            f1s = []
            for label, pred in zip(labels, preds):
                f1 = f1_score(label.cpu(), pred.cpu(), average='macro')
                f1s.append(f1)
            batch_f1 = sum(f1s) / len(f1s)
            metric.update('sample_f1', batch_f1)        

            # Macro F1 score computation
            labels = labels.view(-1)
            preds = preds.view(-1)
            f1 = f1_score(labels.cpu(), preds.cpu(), average='macro')
            metric.update('batch_f1', f1)    

            if debug is True:
                cnt += 1
                if cnt == 2:
                    break

    # Save output and target files
    out_dir = jpath(config['result_root'], config['out_dir'])
    latest_version_dir, ckpt_fp = get_latest_checkpoint(out_dir)
    save_dir = jpath(out_dir, 'lightning_logs', latest_version_dir)
    test_out_fp = jpath(save_dir, '{}_out.txt'.format(split))
    
    data = []
    for tgt, out in zip(tgts, outs):
        tgt_str = ' '.join([str(t) for t in tgt])
        out_str = ' '.join([str(o) for o in out]) 
        data.append('{}\n{}\n'.format(tgt_str, out_str))
    with open(test_out_fp, 'w') as f:
        f.write('\n'.join(data))

    # Save metrics
    res = metric.average()
    save_json(res, jpath(save_dir, '{}_metrics.json'.format(split)))



class Metric:
    def __init__(self):
        self.clear()

    def clear(self):
        self.metrics = {}

    def update(self, metric_name, metric_value, extend=False):
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []

        if not extend:
            self.metrics[metric_name].append(metric_value)
        else:
            self.metrics[metric_name].extend(metric_value)

    def average(self):
        for metric_name in self.metrics:
            t = self.metrics[metric_name]
            self.metrics[metric_name] = sum(t) / len(t)
        ret = self.metrics

        for k in ret:
            ret[k] = round(ret[k], 5) * 100

        self.clear()
        return ret

    



if __name__ == '__main__':
    main()