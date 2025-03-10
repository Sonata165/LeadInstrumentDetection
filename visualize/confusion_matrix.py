'''
Visualize the model's output by drawing the confusion matrix

Author: Longshen Ou, 2024/07/25
'''

import os
import sys

sys.path.append('.')
sys.path.append('..')

import mlconfig
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
from utils import jpath, read_json, save_json, create_if_not_exist, print_json, get_latest_checkpoint
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    if not len(sys.argv) == 2:
        config_fp = '/home/longshen/work/SoloDetection/hparams/mjn_chcls/ch_permute.yaml'
        config = mlconfig.load(config_fp)
        config['train']['fast_dev_run'] = 5
        debug = True
    else:
        config_fp = sys.argv[1]
        config = mlconfig.load(config_fp)
        debug = False

    split = 'valid'  # 'valid' or 'test'
    
    out_dir = jpath(config['result_root'], config['out_dir'])
    latest_version_dir, ckpt_fp = get_latest_checkpoint(out_dir)
    save_dir = jpath(out_dir, 'lightning_logs', latest_version_dir)
    out_fp = jpath(save_dir, '{}_out.txt'.format(split))
    with open(out_fp, 'r') as f:
        data = f.read()
    tgts_and_outs = data.split('\n\n')

    outs = []
    tgts = []
    for tgt_and_out in tgts_and_outs:
        tgt, out = tgt_and_out.strip().split('\n')
        outs.extend(out.split())
        tgts.extend(tgt.split())

    inst_dict = read_json('/home/longshen/work/SoloDetection/misc/inst_id_to_inst_name.json')
    # inst_dict = read_json(jpath('misc', 'inst_id_to_inst_name.json'))

    # Compute confusion matrix
    confusion_matrix = compute_custom_confusion_matrix(outs, tgts, inst_dict)
    # Remove first row and first column
    # confusion_matrix = confusion_matrix[1:, 1:]
    # Prepare inst name from inst_idx_to_name
    inst_names = [inst_dict[str(idx)] for idx in range(1, len(inst_dict)+1)]
    # # Use inst_names from inst_idx_to_name as column and row names
    confusion_matrix = pd.DataFrame(confusion_matrix, columns=inst_names, index=inst_names)
    # Save confusion matrix to file
    pd.DataFrame(confusion_matrix).to_csv(jpath(save_dir, 'confusion_matrix_{}.csv'.format(split)))

    # # Draw a confusion matrix heatmap
    # plt.figure(figsize=(12, 7))
    # sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')  # 加入色彩映射增加可读性
    # plt.xlabel('Predicted Labels')
    # plt.ylabel('True Labels')
    # plt.title('Confusion Matrix Heatmap')  # 添加标题以提供更多信息
    # plt.tight_layout()  # 调整布局以避免标签被遮挡
    # plt.savefig(jpath(out_dir, 'frame_svm_confusion_matrix.png'))
    # plt.show()  # 显示图形，以便在运行脚本时查看结果

    # 应用对数变换到混淆矩阵色彩映射，但使用原始值进行标注
    plt.figure(figsize=(12, 7))
    sns.heatmap(np.log1p(confusion_matrix), annot=confusion_matrix, fmt='d', cmap='Blues', annot_kws={'size': 9})
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix Heatmap (Log Scale Coloring)')  # 添加标题以提供更多信息
    plt.tight_layout()  # 调整布局以避免标签被遮挡
    plt.savefig(jpath(save_dir, 'confusion_matrix_{}.png'.format(split)))


def compute_custom_confusion_matrix(out, tgt, inst_idx_to_name):
    """
    Compute a confusion matrix for given output and target arrays, ensuring all labels
    defined in inst_idx_to_name are included in the matrix.

    Parameters:
    out (np.ndarray): 1D array of integers, predicted labels.
    tgt (np.ndarray): 1D array of integers, true labels.
    inst_idx_to_name (list): List of all possible labels as strings.

    Returns:
    np.ndarray: Confusion matrix with dimensions len(inst_idx_to_name) x len(inst_idx_to_name)
    """
    out = [int(o) for o in out]
    tgt = [int(t) for t in tgt]

    # # Convert labels from string to integer indices
    # label_to_index = {label: idx for idx, label in enumerate(inst_idx_to_name)}
    
    # # Map outputs and targets to corresponding indices
    # out_mapped = np.array([label_to_index[str(label)] for label in out if str(label) in label_to_index])
    # tgt_mapped = np.array([label_to_index[str(label)] for label in tgt if str(label) in label_to_index])
    
    # Create a confusion matrix
    # Initialize a confusion matrix with zeros
    cm = np.zeros((len(inst_idx_to_name), len(inst_idx_to_name)), dtype=int)
    
    # Compute the confusion matrix only for labels that exist in out and tgt
    cm_update = confusion_matrix(tgt, out, labels=range(1, len(inst_idx_to_name)+1))
    
    # Since some labels might not appear in out or tgt, we need to ensure the matrix size is correct
    # This might be redundant depending on whether confusion_matrix fills the shape correctly
    cm[:cm_update.shape[0], :cm_update.shape[1]] = cm_update

    return cm

if __name__ == '__main__':
    main()