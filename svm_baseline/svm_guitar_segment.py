'''
Segment-level SVM for guitar solo detection. Binary classification.
Try to reproduce (Pati, 2017) A Dataset and Method for Electric Guitar Solo Detection in Rock Music

Author: Longshen Ou, 2024/07/25
'''

import os
import sys

sys.path.append('.')
sys.path.append('..')

import msaf
import numpy as np
import torch
import librosa
import torchaudio
import numpy as np
from utils import *
from tqdm import tqdm
from sklearn import svm
from sklearn.metrics import confusion_matrix, f1_score


def main():
    pass


def procedures():
    segment_svm()           # Train the SVM model, get the output
    frame_svm_evaluate()    # Evaluate the SVM's output


def segment_svm():
    '''
    Train a frame-level single-label multi-class classification
    '''
    data_dir = '/home/longshen/data/datasets/MJN/segmented/svm/aggregated'
    train_X = torch.load(jpath(data_dir, 'train_X.pt')) # [n_sample, n_feat]
    train_y = torch.load(jpath(data_dir, 'train_y.pt'))

    # Normalize features
    train_X = (train_X - train_X.mean(dim=0)) / train_X.std(dim=0)

    clf = svm.SVC()

    print('Training in progress ...')
    clf.fit(train_X, train_y)

    # Prepare test sets
    test_splits = ['valid', 'test']
    for test_split in test_splits:
        test_X = torch.load(jpath(data_dir, '{}_X.pt'.format(test_split)))
        test_y = torch.load(jpath(data_dir, '{}_y.pt'.format(test_split)))

        # Normalize features
        test_X = (test_X - test_X.mean(dim=0)) / test_X.std(dim=0)

        print('Predicting ...')
        out = clf.predict(test_X) # ndarray [n_frame,]

        # Save output, y_test, inst_idx_to_name
        out_dir = '/home/longshen/data/datasets/MJN/segmented/svm/out'
        save_fp = jpath(out_dir, '{}_out.json'.format(test_split))
        save_json(out.tolist(), save_fp)
        save_fp = jpath(out_dir, '{}_tgt.json'.format(test_split))
        save_json(test_y.tolist(), save_fp)


def frame_svm_evaluate():
    '''
    Evaluate the frame-level SVM

    Metrics: 
    - Accuacy
    - F1 of each class
    - Macro F1
    - Confusion matrics
    '''
    out_dir = '/home/longshen/data/datasets/MJN/segmented/svm/out'

    splits = ['valid', 'test']
    for split in splits:

        out = read_json(jpath(out_dir, '{}_out.json'.format(split)))
        tgt = read_json(jpath(out_dir, '{}_tgt.json'.format(split)))

        res = {}

        # Evaluate accuracy between out and tgt (two lists of integers)
        out = np.array(out)
        tgt = np.array(tgt)
        acc = np.mean(out == tgt)
        # Round to 5 decimal places
        acc = round(acc, 5)
        res['accuracy'] = acc

        # Evaluate F1 of each class
        f1 = f1_score(tgt, out)
        res['f1'] = f1

        macro_f1 = f1_score(tgt, out, average='macro')
        res['macro_f1'] = macro_f1

        # Draw a confusion matrix for tgt and out, a binary classification task, with matplotlib
        
        import matplotlib.pyplot as plt
        import seaborn as sns
        # plot with sns.heatmap
        cm = confusion_matrix(tgt, out)

        plt.figure(figsize=(9, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

        # 设置正确的标签和位置
        plt.xticks([0.5, 1.5], ['non-guitar', 'guitar-solo'])  # 将标签设置在每个格子的中心
        plt.yticks([0.5, 1.5], ['non-guitar', 'guitar-solo'], va='center')

        plt.xlabel('Prediction')
        plt.ylabel('Ground truth')
        plt.title('Confusion Matrix')

        # 保存图片
        plt.savefig(jpath(out_dir, '{}_confusion_matrix.png'.format(split)))

        # Save the result
        res_fp = jpath(out_dir, '{}_metrics.json'.format(split))
        save_json(res, res_fp)



annot_inst_map = { # map between inst in annotation and stem name
    'eb': 'bass',
    'tr': 'trumpet',
    'dr': 'drum',
    'vo': 'vocal',
    'bt': 'pc',
    'eg': 'guitar'
}


def prepare_baseline_features():
    '''
    Prepare frame-level dataset for SVM
    The dataset is only generated for a given performance
    return: X_train (n_frame_tot, n_feat), y_train (n_samples), X_test, y_test
    '''
    seg_dataset_root = '/home/longshen/data/datasets/MJN/segmented/5perfs'
    meta_fp = jpath(seg_dataset_root, 'metadata.json')
    meta = read_json(meta_fp)
    
    prepare_baseline_features_and_label_of_split(meta, 'train')
    prepare_baseline_features_and_label_of_split(meta, 'test')

    return


def prepare_pitch_features():
    '''
    Prepare frame-level dataset for SVM
    The dataset is only generated for a given performance
    return: X_train (n_frame_tot, n_feat), y_train (n_samples), X_test, y_test
    '''
    seg_dataset_root = '/home/longshen/data/datasets/MJN/segmented/5perfs'
    meta_fp = jpath(seg_dataset_root, 'metadata.json')
    meta = read_json(meta_fp)
    
    prepare_pitch_features_of_split(meta, 'train')
    prepare_pitch_features_of_split(meta, 'test')

    return


def prepare_segment_features():
    '''
    Prepare frame-level dataset for SVM
    The dataset is only generated for a given performance
    return: X_train (n_frame_tot, n_feat), y_train (n_samples), X_test, y_test
    '''
    seg_dataset_root = '/home/longshen/data/datasets/MJN/segmented/5perfs'
    meta_fp = jpath(seg_dataset_root, 'metadata.json')
    meta = read_json(meta_fp)
    
    prepare_baseline_features_and_label_of_split(meta, 'train')
    prepare_baseline_features_and_label_of_split(meta, 'test')

    return


def prepare_baseline_features_and_label_of_split(meta, split):
    '''
    Prepare the segment-level data, consisting extracted segment-level feature, and corresponding label of each segment

    Parameters:
    meta (dict): The metadata of the MJN dataset
    split (str): The split of the dataset
    '''
    data_dir = '/home/longshen/data/datasets/MJN/segmented/5perfs'

    X, y = None, None
    insts = None
    inst_name_to_idx = None
    inst_idx_to_name = None

    Xs = [] # List of segment-level features
    ys = [] # List of segment-level labels, binary. 1 means more than half of the segment is soloing guitar, 0 otherwise

    meta_split = {pfm_seg_id: meta[pfm_seg_id] for pfm_seg_id in meta if meta[pfm_seg_id]['split'] == split}

    for i, pfm_seg_id in enumerate(tqdm(meta_split)):
        t = pfm_seg_id.split('-')
        entry = meta_split[pfm_seg_id]
        seg_split = entry['split']
        
        # Read the lead instrument annotation
        annots = entry['annotations']

        # Obtain the segment-level label
        guitar_solo_seconds = 0
        for annot in annots:
            if 'guitar' in annot['lead']:
                start_time = timecode_to_seconds(annot['start'])
                end_time = timecode_to_seconds(annot['end'])
                guitar_solo_seconds += end_time - start_time
        if guitar_solo_seconds > 2.5: # More than half of the segment is soloing guitar
            y = 1
        else:
            y = 0
        ys.append(y)

        # Read the mixture audio
        audio_dir = jpath(data_dir, entry['seg_audio_dir'])
        audio_fp = jpath(audio_dir, 'mix.mp3')
        audio, sr = torchaudio.load(audio_fp)
        audio = convert_waveform_to_mono(audio) # [1, n_samples]
        
        # Resample to 44100Hz
        tgt_sr = 44100
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=tgt_sr)
        audio = resampler(audio)

        ''' Feature Extraction '''
        baseline_features = extract_baseline_features(audio, sr=tgt_sr) # [n_frame, 17]

        Xs.append(baseline_features)

        # predominant_pitch_features = extract_predominant_pitch_features(audio, sr=tgt_sr)
        # structural_segment_features = extract_structural_segment_features(audio, sr=tgt_sr)

        # # Segment audio to 2s segments, 1s overlap, 4 segments in total
        # # Extract features for each segment
        # audio_segs = []
        # for i in range(4):
        #     start = i * 1
        #     end = (i + 2) * 1
        #     audio_seg = audio[:, start * tgt_sr: end * tgt_sr]
        #     features = extract_baseline_features(audio_seg, sr=tgt_sr)
        #     Xs.append(features)

        # # 21 dim in total: 17 baseline feature, 2 pitch feature, 2 structural feature
        

        # # Aggregation: a list of dim=21 features -> mean and std of them, dim=42
        # X_2s = np.zeros(shape=(4, 21))
        # X_mean = X_2s.mean(axis=0)
        # X_std = X_2s.std(axis=0)
        # X = np.concatenate((X_mean, X_std)) # dim=42

    Xs = torch.stack(Xs) # [n_samples, n_frame_tot, 17]
    ys = torch.tensor(ys) # [n_samples]    

    # Save to data dir
    save_root = '/home/longshen/data/datasets/MJN/segmented/svm'
    split_dir = jpath(save_root, split)
    create_if_not_exist(split_dir)
    save_fn = jpath(split_dir, 'features_baseline.pt')
    torch.save(Xs, save_fn)
    save_fn = jpath(split_dir, 'labels.pt')
    torch.save(ys, save_fn)

    # return Xs, ys
    return


def prepare_pitch_features_of_split(meta, split):
    '''
    Prepare the segment-level data, consisting extracted segment-level feature, and corresponding label of each segment

    Parameters:
    meta (dict): The metadata of the MJN dataset
    split (str): The split of the dataset
    '''
    data_dir = '/home/longshen/data/datasets/MJN/segmented/5perfs'

    X, y = None, None
    insts = None
    inst_name_to_idx = None
    inst_idx_to_name = None

    Xs = [] # List of segment-level features
    ys = [] # List of segment-level labels, binary. 1 means more than half of the segment is soloing guitar, 0 otherwise

    meta_split = {pfm_seg_id: meta[pfm_seg_id] for pfm_seg_id in meta if meta[pfm_seg_id]['split'] == split}

    for i, pfm_seg_id in enumerate(tqdm(meta_split)):
        t = pfm_seg_id.split('-')
        entry = meta_split[pfm_seg_id]
        seg_split = entry['split']
        
        # Read the mixture audio
        audio_dir = jpath(data_dir, entry['seg_audio_dir'])
        audio_fp = jpath(audio_dir, 'mix.mp3')
        audio, sr = torchaudio.load(audio_fp)
        audio = convert_waveform_to_mono(audio) # [1, n_samples]
        
        # Resample to 44100Hz
        tgt_sr = 44100
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=tgt_sr)
        audio = resampler(audio)

        ''' Feature Extraction '''
        pitch_features = extract_predominant_pitch_features(audio, sr=tgt_sr) # [n_frame,]
        # Convert feature to midi pitch
        Xs.append(pitch_features)


    # pad Xs to longest frame size
    max_frame_size = max([X.shape[0] for X in Xs])
    for i in range(len(Xs)):
        Xs[i] = torch.nn.functional.pad(Xs[i], (0, max_frame_size - Xs[i].shape[0]))

    Xs = torch.stack(Xs) # [n_samples, n_frame_tot, 2]

    # Save to data dir
    save_root = '/home/longshen/data/datasets/MJN/segmented/svm'
    split_dir = jpath(save_root, split)
    create_if_not_exist(split_dir)
    save_fn = jpath(split_dir, 'features_pitch.pt')
    torch.save(Xs, save_fn)

    return


def prepare_segmentation_features_for_songs():
    '''
    Prepare the segment-level data, consisting extracted segment-level feature, and corresponding label of each segment

    Parameters:
    meta (dict): The metadata of the MJN dataset
    split (str): The split of the dataset
    '''
    data_dir = '/home/longshen/data/datasets/MJN/mix_audio'
    save_dir = '/home/longshen/work/SoloDetection/svm_baseline/segmentation_outs'

    audio_fns = ls(data_dir)
    for audio_fn in tqdm(audio_fns):
        audio_fp = jpath(data_dir, audio_fn)

        ''' Feature Extraction '''
        save_fp = jpath(save_dir, audio_fn.replace('.wav', '.txt'))
        extract_structural_segment_features(audio_fp, save_fp) # [n_frame, 17]

    return


def prepare_segmentation_features_of_split(meta, split):
    '''
    Prepare the segment-level data, consisting extracted segment-level feature, and corresponding label of each segment

    Parameters:
    meta (dict): The metadata of the MJN dataset
    split (str): The split of the dataset
    '''
    data_dir = '/home/longshen/data/datasets/MJN/mix_audio'
    save_dir = '/home/longshen/work/SoloDetection/svm_baseline/segmentation_outs'

    audio_fns = ls(data_dir)
    for audio_fn in audio_fns:
        audio_fp = jpath(data_dir, audio_fn)

        ''' Feature Extraction '''
        save_fp = jpath(save_dir, audio_fn.replace('.mp3', '.txt'))
        extract_structural_segment_features(audio_fp, save_fp) # [n_frame, 17]

    return



# 使用librosa计算零交叉率
def zero_crossing_rate_librosa(audio, frame_length, hop_length):
    return librosa.feature.zero_crossing_rate(audio, frame_length=frame_length, hop_length=hop_length)[0]

# 更新提取特征函数中的zero_crossing_rate调用
def extract_baseline_features(audio, sr=44100, n_fft=2048, hop_length=1024):
    # 确保音频是单通道
    if audio.ndim > 1:
        audio = audio.mean(dim=0)  # 若是多通道，取平均值合成单通道, shape: [n_samples,]

    audio_tensor = audio
    audio_np = audio.numpy()

    # 展开音频张量为帧
    framed_audio = audio_tensor.unfold(0, n_fft, hop_length)

    # 1. 频谱质心
    spectral_centroid = torchaudio.transforms.SpectralCentroid(sample_rate=sr, n_fft=n_fft, hop_length=hop_length)(audio_tensor.unsqueeze(0)) # [1, n_frame]

    # 2. 每帧的最大振幅
    max_amplitude = framed_audio.abs().max(dim=1).values    # [n_frame,]

    # 3. 零交叉率 - 使用librosa
    zero_crossing_rate = torch.tensor(zero_crossing_rate_librosa(audio_np, frame_length=n_fft, hop_length=hop_length)) # [n_frame,]

    # 4. MFCCs (2nd to 13th)
    mfcc_transform = torchaudio.transforms.MFCC(sample_rate=sr, n_mfcc=13, melkwargs={'n_fft': n_fft, 'hop_length': hop_length})
    mfccs = mfcc_transform(audio_tensor.unsqueeze(0))[0, 1:]  # Skip the first MFCC   [12, n_frame]

    # 5. Spectral crest factor
    S = np.abs(librosa.stft(audio_np, n_fft=n_fft, hop_length=hop_length))   # [n_frame,]
    spectral_crest = torch.tensor(np.max(S, axis=0) / np.mean(S, axis=0))

    # 6. Spectral flux
    onset_env = librosa.onset.onset_strength(y=audio_np, sr=sr, hop_length=hop_length)
    spectral_flux = torch.tensor(np.diff(onset_env, prepend=onset_env[0]))    # [n_frame,]

    spectral_centroid = spectral_centroid.squeeze(0)
    mfccs = torch.permute(mfccs, (1, 0)) # [n_frame, 12]
    
    # pad to the longest frame size
    max_frame_size = max([spectral_centroid.shape[0], max_amplitude.shape[0], zero_crossing_rate.shape[0], mfccs.shape[0], spectral_crest.shape[0], spectral_flux.shape[0]])
    spectral_centroid = torch.nn.functional.pad(spectral_centroid, (0, max_frame_size - spectral_centroid.shape[0]))
    max_amplitude = torch.nn.functional.pad(max_amplitude, (0, max_frame_size - max_amplitude.shape[0]))
    zero_crossing_rate = torch.nn.functional.pad(zero_crossing_rate, (0, max_frame_size - zero_crossing_rate.shape[0]))
    mfccs = torch.nn.functional.pad(mfccs, (0, 0, 0, max_frame_size - mfccs.shape[0]))
    spectral_crest = torch.nn.functional.pad(spectral_crest, (0, max_frame_size - spectral_crest.shape[0]))
    spectral_flux = torch.nn.functional.pad(spectral_flux, (0, max_frame_size - spectral_flux.shape[0]))

    # Concat along feature dim
    features = torch.stack((spectral_centroid, max_amplitude, zero_crossing_rate, spectral_crest, spectral_flux), dim=1)
    features = torch.cat((features, mfccs), dim=1)

    return features # [n_frame, 17]


def extract_predominant_pitch_features(audio, sr):
    import crepe
    time, frequency, confidence, activation = crepe.predict(audio.squeeze().numpy(), sr, viterbi=True, step_size=23)
    pitch = librosa.hz_to_midi(frequency)

    # Concat frequency and confidence on a new dim
    pitch = torch.tensor(pitch)
    confidence = torch.tensor(confidence)
    features = torch.stack((pitch, confidence), dim=1) # [n_frame, 2]

    return features # [n_frame, 2] value: float, Hz


def extract_structural_segment_features(audio_fp, out_fp):
    boundaries, labels = msaf.process(audio_fp, boundaries_id='cnmf', labels_id='scluster')
    msaf.io.write_mirex(boundaries, labels, out_fp)


if __name__ == '__main__':
    main()