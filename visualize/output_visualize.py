'''
Visualize the model's output by drawing the multitrack heatmap together with label and predictions

Author: Longshen Ou, 2024/07/25
'''

import os
import sys

sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from utils import jpath, create_if_not_exist
import seaborn as sns
from random import sample
from lightning_train import get_dataloader
import mlconfig
from inst_utils import InstTokenizer
from tqdm import tqdm
# from line_profiler import LineProfiler

# 可以根据需要修改的标签名称
LABEL_NAMES = ['Guitar', 'Vocal', 'Drums', 'Bass', 'Keyboard', 'Violin', 'Sax', 'Flute', 'Clarinet', 'Trumpet']

def main():
    # lp = LineProfiler()
    # lp_wrapper = lp(test_visualize_output)
    # test_visualize_output()
    # lp.print_stats()

    test_visualize_output()


def test_visualize_output():
    out_fp = '/home/longshen/data/results/mjn_5pfm/chcls/chattn/ch_permute/bs16_drop0.8/lightning_logs/version_0/test_out.txt'
    config = mlconfig.load('/home/longshen/work/SoloDetection/hparams/mjn_chcls/ch_permute.yaml')
    test_loader = get_dataloader(config, split='test', bs=1)

    # Read output file
    with open(out_fp) as f:
        data = f.read()
    ref_out_pairs = data.split('\n\n')

    save_root = os.path.dirname(out_fp)
    save_dir = jpath(save_root, 'visualize')
    create_if_not_exist(save_dir)
    
    tk = InstTokenizer()
    label_dict = {i: tk.convert_id_to_inst(i) for i in range(1, tk.vocab_size() + 1)}

    cnt = 0
    for id, (batch, ref_out_pairs) in enumerate(tqdm(zip(test_loader, ref_out_pairs), total=len(test_loader))):
        pass
        audios = batch['audio'][0].cpu().numpy() # [max_n_inst, n_samples=120320]
        energy = frame_level_sqrt(audios)[:,:-1]
        audios = energy
        inst_ids = batch['inst_ids'][0].tolist() # [max_n_insts]
        label, out = ref_out_pairs.split('\n')
        label = np.array([int(i) for i in label.strip().split(' ')])
        out = np.array([int(i) for i in out.strip().split(' ')])
        inst_of_each_channel = [tk.convert_id_to_inst(i) for i in inst_ids]

        # Only continue when label contains more than 1 instrument
        if len(set(label)) < 2:
            continue

        # Put the 'mix' channel to the first row
        mix_idx = inst_of_each_channel.index('mix')
        inst_of_each_channel[0], inst_of_each_channel[mix_idx] = inst_of_each_channel[mix_idx], inst_of_each_channel[0]
        mix_audio = audios[mix_idx].copy()
        ch0_audio = audios[0].copy()
        audios[0] = mix_audio
        audios[mix_idx] = ch0_audio

        save_fp = jpath(save_dir, f'output_visualize_{id}.png')
        visualize_output(audios, inst_of_each_channel, out, label, save_fp, label_dict) # slow

        cnt += 1


def frame_level_sqrt(audios: np.ndarray, frame_size=320):
    # audios should be [max_n_inst, n_samples=120320]
    # Initialize the list to store frame energies
    energies = np.zeros((audios.shape[0], audios.shape[1] // frame_size))
    
    # Loop through each instance
    for idx in range(audios.shape[0]):
        # Process each frame
        for i in range(0, audios.shape[1] - frame_size + 1, frame_size):
            frame = audios[idx, i:i+frame_size]
            # Calculate the energy of the frame
            energy = np.sum(frame**2)
            energy = np.sqrt(energy)
            energies[idx, i // frame_size] = energy
    
    return energies

def demo():
    num_channels = 6  # 定义音频通道数
    num_frames = 375
    audio_data = np.random.rand(num_channels, num_frames)
    audio_channel_insts = sample(LABEL_NAMES, k=num_channels)
    predicted_labels = generate_labels(num_frames, 10, duration=150)
    true_labels = generate_labels(num_frames, 10, duration=150)

    save_dir = '/home/longshen/work/SoloDetection/misc'
    save_fp = jpath(save_dir, 'output_visualize.png')
    
    visualize_output(audio_data, audio_channel_insts, predicted_labels, true_labels, save_fp)

def generate_labels(num_frames, num_labels, duration=150):
    labels = np.zeros(num_frames, dtype=int)
    current_label = np.random.randint(1, num_labels + 1)
    for start in range(0, num_frames, duration):
        end = min(start + duration, num_frames)
        labels[start:end] = current_label
        current_label = np.random.randint(1, num_labels + 1)
    return labels

def visualize_output(audio_data, 
                     audio_channel_insts, 
                     predicted_labels: np.ndarray, 
                     true_labels: np.ndarray, 
                     save_fp,
                     label_dict: dict
                     ):
    num_frames = predicted_labels.shape[0]
    # labels_dict = {i+1: LABEL_NAMES[i] for i in range(len(LABEL_NAMES))}  # 动态生成标签字典
    
    colors = list(mcolors.TABLEAU_COLORS.values())  # 颜色列表

    fig, axs = plt.subplots(3, 1, figsize=(10, 5), gridspec_kw={'height_ratios': [4, 0.33, 0.33]})

    # 使用Seaborn绘制热图，并设置乐器名称为y轴标签
    
    # axs[0].set_title('Multi-channel Audio Data')
    axs[0].set_ylabel('Instrument')
    axs[0].xaxis.tick_top()
    axs[0].xaxis.set_label_position('top')
    sns.heatmap(audio_data, ax=axs[0], cbar=True, cmap='viridis', cbar_kws={'location': 'top', 'shrink': 0.6, 'pad': 0.15},
                yticklabels=audio_channel_insts)

    plot_labels(axs[1], true_labels, colors, 'True Labels', label_dict)
    plot_labels(axs[2], predicted_labels, colors, 'Predicted Labels', label_dict)

    fig.subplots_adjust(hspace=0.2)  # 调整子图之间的垂直间距

    # plt.tight_layout()
    plt.savefig(save_fp)

def plot_labels(ax, labels, colors, title, labels_dict):
    ax.set_xlim([0, len(labels)])
    ax.set_ylim([0, 1])
    last_label = labels[0]
    i_prev = 0

    n_labels = len(set(labels))
    for i in range(1, len(labels)):
        if labels[i] != last_label:
            ax.axvspan(i_prev, i, color=colors[(last_label - 1) % len(colors)], alpha=0.5)
            ax.text((i_prev + i) / 2, 0.5, labels_dict.get(last_label, 'Unknown'), ha='center', va='center', fontsize=12, color='black', fontweight='bold')
            i_prev = i
        last_label = labels[i]
    ax.axvspan(i_prev, len(labels), color=colors[(last_label - 1) % len(colors)], alpha=0.5)
    ax.text((i_prev + len(labels)) / 2, 0.5, labels_dict.get(last_label, 'Unknown'), ha='center', va='center', fontsize=12, color='black', fontweight='bold')

    ax.set_title(title, pad=1)
    ax.set_yticks([])
    ax.set_xticks([])


if __name__ == '__main__':
    main()