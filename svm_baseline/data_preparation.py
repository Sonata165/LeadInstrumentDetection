'''
Prepare features and labels for the SVM baseline

Author: Longshen Ou, 2024/07/25
'''

import os
import sys

sys.path.append('.')
sys.path.append('..')

import msaf
import torch
import librosa
import torchaudio
import numpy as np
from utils import *
from tqdm import tqdm
from sklearn.metrics import confusion_matrix


def main():
    aggregate_features()


def procedures():
    prepare_segmentation_features_for_songs()
    prepare_baseline_features()
    prepare_pitch_features()
    prepare_segment_features()
    aggregate_features()


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
    # Convert labels from string to integer indices
    label_to_index = {label: idx for idx, label in enumerate(inst_idx_to_name)}
    
    # Map outputs and targets to corresponding indices
    out_mapped = np.array([label_to_index[str(label)] for label in out if str(label) in label_to_index])
    tgt_mapped = np.array([label_to_index[str(label)] for label in tgt if str(label) in label_to_index])
    
    # Create a confusion matrix
    # Initialize a confusion matrix with zeros
    cm = np.zeros((len(inst_idx_to_name), len(inst_idx_to_name)), dtype=int)
    
    # Compute the confusion matrix only for labels that exist in out and tgt
    cm_update = confusion_matrix(tgt_mapped, out_mapped, labels=range(len(inst_idx_to_name)))
    
    # Since some labels might not appear in out or tgt, we need to ensure the matrix size is correct
    # This might be redundant depending on whether confusion_matrix fills the shape correctly
    cm[:cm_update.shape[0], :cm_update.shape[1]] = cm_update

    return cm


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
    prepare_baseline_features_and_label_of_split(meta, 'valid')

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
    prepare_pitch_features_of_split(meta, 'valid')

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
    
    prepare_segmentation_features_of_split(meta, 'train')
    prepare_segmentation_features_of_split(meta, 'test')
    prepare_segmentation_features_of_split(meta, 'valid')

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
    song_seg_info_dir = '/home/longshen/work/SoloDetection/svm_baseline/segmentation_outs' # dir to save song-level segment info

    Xs = [] # List of segment-level features

    meta_split = {pfm_seg_id: meta[pfm_seg_id] for pfm_seg_id in meta if meta[pfm_seg_id]['split'] == split}

    for i, pfm_seg_id in enumerate(tqdm(meta_split)):
        t = pfm_seg_id.split('-')
        pfm_id = '-'.join(t[:2])

        # Read the song-level segment info
        song_seg_info_fp = jpath(song_seg_info_dir, f'{pfm_id}.txt')
        sec_to_dur, sec_to_rep = parse_segment_info(song_seg_info_fp)

        entry = meta_split[pfm_seg_id]
        start_time = timecode_to_seconds(entry['original_start'])
        end_time = timecode_to_seconds(entry['original_end'])

        # Construct frame-level features, 43 Hz
        
        # Divide the span from start_time to endtime to 1/43 s segment
        features = []
        for sec in np.arange(start_time, end_time, 1/43):
            dur = sec_to_dur[sec]
            rep = sec_to_rep[sec]
            features.append([dur, rep])
        
        features = torch.tensor(features) # [n_frame, 2]

        Xs.append(features)


    # pad Xs elements to longest frame size (the first dim)
    max_frame_size = max([X.shape[0] for X in Xs])
    for i in range(len(Xs)):
        Xs[i] = torch.nn.functional.pad(Xs[i], (0, 0, 0, max_frame_size - Xs[i].shape[0]))

    Xs = torch.stack(Xs) # [n_samples, n_frame, 2]

    # Save to data dir
    save_root = '/home/longshen/data/datasets/MJN/segmented/svm'
    split_dir = jpath(save_root, split)
    create_if_not_exist(split_dir)
    save_fn = jpath(split_dir, 'features_structure.pt')
    torch.save(Xs, save_fn)

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


def parse_segment_info(song_seg_info_fp):
    '''
    Read a song-level segment info file, parse to two float_dict

    - Convert a time in second to corresponding segment's duration
    - Convert a time in second to corresponding segment's label repetition
    
    '''
    sec_to_seg_id = {}
    seg_id_to_dur = {}
    seg_id_to_label = {}
    label_to_repetition = {}
    
    with open(song_seg_info_fp, 'r') as f:
        lines = f.readlines()
    seg_id = 0
    for line in lines:
        start, end, label = line.strip().split()
        start_sec = float(start)
        end_sec = float(end)
        label = int(float(label))
        
        sec_to_seg_id[end] = seg_id
        seg_id_to_label[seg_id] = label
        seg_id_to_dur[seg_id] = end_sec - start_sec

        if label not in label_to_repetition:
            label_to_repetition[label] = 0
        label_to_repetition[label] += 1

        seg_id += 1

    # Dur_of_seg convert to float_dict
    sec_to_seg_id = float_dict.from_dict(sec_to_seg_id)

    # Get song length
    song_length = float(lines[-1].split()[1])

    # Convert seg_id_to_dur's value to relative duration
    for seg_id in seg_id_to_dur:
        seg_id_to_dur[seg_id] /= song_length

    # Make a float_dict: sec_to_dur
    keys = sec_to_seg_id.keys
    values = [seg_id_to_dur[i] for i in sec_to_seg_id.values]
    sec_to_dur = float_dict(keys, values)

    # Make a float_dict: sec_to_label_repetition
    keys = sec_to_seg_id.keys
    values = [label_to_repetition[seg_id_to_label[i]] for i in sec_to_seg_id.values]
    sec_to_label_repetition = float_dict(keys, values)

    return sec_to_dur, sec_to_label_repetition


def aggregate_features():
    '''
    Create train_X, train_y, test_X, test_y from the segmented features
    Save to pt files

    Samples are in segment-level
    By compute the mean and std of all the frame over the entire segment
    '''
    data_dir = '/home/longshen/data/datasets/MJN/segmented/svm'
    out_dir = '/home/longshen/data/datasets/MJN/segmented/svm/aggregated'

    for split in ['valid', 'test', 'train']:
        split_dir = jpath(data_dir, split)
        
        features_baseline = torch.load(jpath(split_dir, 'features_baseline.pt'))  # [n_samples, n_frame_tot, 17]
        features_pitch = torch.load(jpath(split_dir, 'features_pitch.pt')) # [n_samples, n_frame_tot, 2]
        features_structure = torch.load(jpath(split_dir, 'features_structure.pt')) # [n_samples, n_frame_tot, 2]
        labels = torch.load(jpath(split_dir, 'labels.pt')) # [n_samples,]

        # Ensure they have the same n_samples
        assert features_baseline.shape[0] == features_pitch.shape[0] == features_structure.shape[0] == labels.shape[0]

        # Calculate the mean and std over the frame dim
        features_baseline_mean = features_baseline.mean(dim=1) # [n_samples, 17]
        features_baseline_std = features_baseline.std(dim=1)
        features_pitch_mean = features_pitch.mean(dim=1) # [n_samples, 2]
        features_pitch_std = features_pitch.std(dim=1)
        features_structure_mean = features_structure.mean(dim=1) # [n_samples, 2]
        features_structure_std = features_structure.std(dim=1)

        # Concatenate along the feature dim
        features = torch.cat((features_baseline_mean, features_baseline_std, features_pitch_mean, features_pitch_std, features_structure_mean, features_structure_std), dim=1)
        
        # Save to out_dir
        out_fn_feat = '{}_X.pt'.format(split)
        out_fn_label = '{}_y.pt'.format(split)
        out_feat_fp = jpath(out_dir, out_fn_feat)
        out_label_fp = jpath(out_dir, out_fn_label)
        torch.save(features, out_feat_fp)
        torch.save(labels, out_label_fp)



if __name__ == '__main__':
    main()