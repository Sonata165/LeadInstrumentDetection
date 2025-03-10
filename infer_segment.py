'''
Inference script for the segment-level guitar solo classification task

Usage: python infer_segment.py /path/to/hparam/file.yaml
Author: Longshen Ou, 2024/07/25
'''

import sys
import torch
import mlconfig 
from tqdm import tqdm
from solo_datasets import *
from lightning_model import *
from lightning_train import get_dataloader
from sklearn.metrics import confusion_matrix
from lightning.pytorch import seed_everything

if __name__ == '__main__':
    seed_everything(42)

torch.backends.cuda.matmul.allow_tf32 = True


def main():
    if not len(sys.argv) == 2:
        config_fp = '/home/longshen/work/SoloDetection/hparams/mjn_instcls/mjn_guitar_seg.yaml'
        config = mlconfig.load(config_fp)
        config['train']['fast_dev_run'] = 5
        debug = True
    else:
        config_fp = sys.argv[1]
        config = mlconfig.load(config_fp)
        debug = False

    # Load a lightning model from checkpoint
    lit_model = load_lit_model(config)
    model = lit_model.model
    model.eval() # class: MuseCocoLMHeadModel
    model.cuda()

    # Prepare the test set
    split = 'test' # 'valid' or 'test'
    test_loader = get_dataloader(config, split=split)

    # Iterate over test set, do the classification, save the output
    outs = []
    tgts = []
    cnt = 0
    with torch.no_grad():
        pbar = tqdm(test_loader)
        for id, batch in enumerate(pbar):
            pbar.set_description(str(id))

            audios = batch['audio'].cuda()
            labels = batch['label']
            out = model(audios).cpu()  # [B, n_frame, n_inst_type=15]

            preds = out.argmax(dim=1) # [B, n_frame]

            # Save the output and target
            outs.extend(preds.tolist())
            tgts.extend(labels.tolist())

            if debug is True:
                cnt += 1
                if cnt == 2:
                    break

    # Draw confusion matrix
    cm = confusion_matrix(tgts, outs)
    # cm = [[499, 22], [38, 210]]

    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

    # Set labels and their positions
    plt.xticks([0.5, 1.5], ['non-guitar', 'guitar-solo'])  # Put label to the center of each tick pair
    plt.yticks([0.5, 1.5], ['non-guitar', 'guitar-solo'], va='center')

    plt.xlabel('Prediction')
    plt.ylabel('Ground truth')
    plt.title('Confusion Matrix')

    # Save figure
    plt.savefig('/home/longshen/data/results/mjn_5pfm/instcls/from_mix_guitar_segment/lightning_logs/version_4/confusion_matrix.png')


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