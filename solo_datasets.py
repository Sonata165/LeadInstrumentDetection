'''
Dataset class for training solo detection models

Author: Longshen Ou, 2024/07/25
'''

import torch
import random
import inst_utils
import torchaudio
import inst_utils_cross_dataset as inst_utils_cd
from utils import read_json, jpath, ls, convert_waveform_to_mono, timecode_to_seconds


def get_dataloader(config, split, bs=None, num_workers=None):
    dataset_cls = config.dataset.dataset_class if 'dataset_class' in config.dataset else 'MJNDataset'
    dataset_cls = eval(dataset_cls)
    dataset = dataset_cls(config.dataset, split=split)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=bs if bs is not None else config.dataset.batch_size, 
        shuffle=True if split == 'train' else False,
        collate_fn=dataset.collate_fn,
        num_workers=config.dataset.num_workers if num_workers is None else num_workers,
    )
    return dataloader


class MJNDataset(torch.utils.data.Dataset):
    '''
    Dataset class for training MERT model for solo detection (frame-level classification)

    For Instrument Classification model

    '''
    def __init__(self, dataset_config, split, tgt_sr=24000):
        self.config = dataset_config

        self.root = dataset_config.data_root
        self.meta_fp = jpath(self.root, dataset_config.meta_fn)

        # self.audio_processor = audio_processor
        self.tgt_sr = dataset_config.tgt_sr
        self.split = split

        self.meta = read_json(self.meta_fp)
        self.meta = {k: v for k, v in self.meta.items() if v['split'] == self.split}

        # Re-index self.meta with key start from 0, original key as a new field 'id'
        for i, v in self.meta.items():
            v['id'] = i
        self.meta = {i: v for i, v in enumerate(self.meta.values())}

        if self.config.get('cross_dataset_tokenizer', False):
            self.inst_tokenizer = inst_utils_cd.InstTokenizer()
        else:
            self.inst_tokenizer = inst_utils.InstTokenizer()
        
    def __getitem__(self, idx):
        '''
        Return 
        - a 5-s clip of multi-channel audios. Each channel is a different instrument.
        - Instrument of each channel
        - Label of the clip
        '''
        sample_meta = self.meta[idx]

        audio_dirpath = jpath(self.root, sample_meta['seg_audio_dir'])
        audio_fns = ls(audio_dirpath)

        # Read audios, convert to mono, stack together
        audio_fps = [jpath(audio_dirpath, audio_fn) for audio_fn in audio_fns]
        audios = [torchaudio.load(audio_fp)[0] for audio_fp in audio_fps]

        # Resample to target sampling rate
        resampler = torchaudio.transforms.Resample(orig_freq=self.config.src_sr, new_freq=self.tgt_sr)
        audios = [resampler(audio) for audio in audios]

        # Right side zero pad 320 samples for each audio
        audios = [torch.nn.functional.pad(audio, (0, 320)) for audio in audios]
        audios = [convert_waveform_to_mono(audio) for audio in audios] # each element: [1, n_samples=120320]
        audio_stack = torch.cat(audios, dim=0) # [n_inst, n_samples=120320]

        # Get instrument names of each channel
        channel_names = [audio_fn.split('.')[0] for audio_fn in audio_fns] # normalized stem names
        if self.config.get('cross_dataset_tokenizer', False):
            inst_names = [inst_utils_cd.from_stem_name_get_inst_name(channel_name) for channel_name in channel_names]
        else:
            inst_names = [inst_utils.from_stem_name_get_inst_name(channel_name) for channel_name in channel_names]
        inst_ids = torch.tensor([self.inst_tokenizer.convert_inst_to_id(inst_name) for inst_name in inst_names], dtype=torch.int64)

        # Process annotations to be frame-level (75Hz)
        annotations = sample_meta['annotations']
        n_frames = self.config.label_frame_rate * self.config.segment_length
        frame_label = torch.zeros(size=(n_frames,), dtype=torch.int64)
        for annot_entry in annotations:
            start_frame = int(timecode_to_seconds(annot_entry['start']) * self.config.label_frame_rate)
            end_frame = int(timecode_to_seconds(annot_entry['end']) * self.config.label_frame_rate)
            lead_inst_stem_name = annot_entry['lead']
            if self.config.get('cross_dataset_tokenizer', False):
                lead_inst_name = inst_utils_cd.from_stem_name_get_inst_name(lead_inst_stem_name)
            else:
                lead_inst_name = inst_utils.from_stem_name_get_inst_name(lead_inst_stem_name)
            lead_inst_id = self.inst_tokenizer.convert_inst_to_id(lead_inst_name)
            frame_label[start_frame:end_frame] = lead_inst_id

        ret = {
            'audio_stack': audio_stack, # tensor of shape [n_inst, n_samples=120320]
            'inst_ids': inst_ids, # tensor of shape [n_inst]
            'label': frame_label, # tensor of shape [n_frames]
        }

        return ret

    def __len__(self):
        return len(self.meta)
    

    def collate_fn(self, batch):
        '''
        Collate function for MJNDataset

        if a batch contains different instruments, inst dim will be padded with zeros to the audio with most instruments
        
        Output format: 
        {
            audio: [bs, n_inst, n_samples=120320]
            inst_ids: [bs, n_inst]  value: one of instrument id
            label: [bs, n_frames]   valud: [0 ~ n_inst], 0: silent; 1 ~ n_inst: channel
        }
            
        '''
        audio = []
        inst_ids = []
        label = []
        for sample in batch:
            audio.append(sample['audio_stack'])
            inst_ids.append(sample['inst_ids'])
            label.append(sample['label'])

        # Ensure all audio have the same number of samples
        n_samples = self.config.segment_length * self.config.tgt_sr + int(self.config.tgt_sr / self.config.label_frame_rate)
        # audio in shape [bs, n_inst, n_samples=120320]
        # Enumerate all audio
        for i, audio_stack in enumerate(audio): 
            # If audio is shorter, pad with zeros
            if audio_stack.size(1) < n_samples:
                pad = torch.zeros(size=(audio_stack.size(0), n_samples - audio_stack.size(1)))
                audio[i] = torch.cat([audio_stack, pad], dim=1)
            # If audio is longer, truncate
            elif audio_stack.size(1) > n_samples:
                audio[i] = audio_stack[:, :n_samples]

        # Pad audio to the same number of instruments
        n_insts = [audio_i.size(0) for audio_i in audio]
        max_n_insts = max(n_insts)
        # ch_pad_masks = []
        for i, audio_stack in enumerate(audio):
            if audio_stack.size(0) < max_n_insts:
                pad = torch.zeros(size=(max_n_insts - audio_stack.size(0), audio_stack.size(1)))
                audio[i] = torch.cat([audio_stack, pad], dim=0)
            # ch_pad_mask = torch.zeros(size=max_n_insts, dtype=torch.bool)
            # ch_pad_mask[audio_stack.size(0):] = True # [max_n_insts]
            # ch_pad_masks.append(ch_pad_mask)
        audio = torch.stack(audio, dim=0) # [bs, max_n_inst, n_samples=120320]
        # ch_pad_masks = torch.stack(ch_pad_masks, dim=0) # [bs, max_n_insts]

        # Pad inst_ids to the same number of instruments
        
        for i, inst_ids_i in enumerate(inst_ids):
            if inst_ids_i.size(0) < max_n_insts:
                pad = torch.zeros(size=(max_n_insts - inst_ids_i.size(0),), dtype=torch.int64)
                inst_ids[i] = torch.cat([inst_ids_i, pad], dim=0)
        inst_ids = torch.stack(inst_ids, dim=0) # [bs, max_n_insts]

        # Collate label
        label = torch.stack(label, dim=0) # [bs, n_frames]

        ret = {
            'audio': audio,   # [bs, max_n_inst, n_samples=120320]
            # 'ch_pad_masks': ch_pad_masks,
            'inst_ids': inst_ids,  # [bs, max_n_insts]
            'label': label,   # [bs, n_frames]
        }

        return ret
    
class MJNDatasetCh(MJNDataset):
    '''
    MJNDataset for channel-wise classification

    For Channel Classification model
    '''

    def __getitem__(self, idx):
        '''
        Return 
        - audios: a 5-s clip of multi-channel audios. Each channel is a different instrument.
        - inst ids: Instrument of each channel
        - labels: Label of the clip. 0: silent; 1 ~ n_inst: channel
        '''
        sample_meta = self.meta[idx]

        audio_dirpath = jpath(self.root, sample_meta['seg_audio_dir'])
        audio_fns = ls(audio_dirpath)

        # Dataset specific setting
        if 'mix.mp3' in audio_fns: # MJN dataset
            mix_fn = 'mix.mp3'
        else:   # MedleyDB dataset
            mix_fn = '0#mix.mp3'
        
        audio_fns.remove(mix_fn)
        if self.split == 'train':
            # Delete the suffix of the audio filenames
            ch_names = [audio_fn.split('.')[0] for audio_fn in audio_fns]

            # Channel permutation augmentation
            if self.config.get('ch_permute_aug', False):
                random.shuffle(ch_names)

            # Recover the audio filenames
            audio_fns = [ch_name + '.mp3' for ch_name in ch_names]
        # Ensure the mix is the first channel
        audio_fns.insert(0, mix_fn)

        # Read audios, convert to mono, stack together
        audio_fps = [jpath(audio_dirpath, audio_fn) for audio_fn in audio_fns]
        audios = [torchaudio.load(audio_fp)[0] for audio_fp in audio_fps]

        # Resample to target sampling rate
        resampler = torchaudio.transforms.Resample(orig_freq=self.config.src_sr, new_freq=self.tgt_sr)
        audios = [resampler(audio) for audio in audios]

        # Right side zero pad 320 samples for each audio
        audios = [torch.nn.functional.pad(audio, (0, 320)) for audio in audios]
        audios = [convert_waveform_to_mono(audio) for audio in audios]
        audio_stack = torch.cat(audios, dim=0) # [n_inst, n_samples=120320]

        # Get channel names of each channel
        channel_names = [audio_fn.split('.')[0] for audio_fn in audio_fns] # normalized stem names
        if self.config.get('cross_dataset_tokenizer', False):
            inst_names = [inst_utils_cd.from_stem_name_get_inst_name(channel_name) for channel_name in channel_names]
        else:
            inst_names = [inst_utils.from_stem_name_get_inst_name(channel_name) for channel_name in channel_names]
        inst_ids = torch.tensor([self.inst_tokenizer.convert_inst_to_id(inst_name) for inst_name in inst_names], dtype=torch.int64)
        channel_names = ['na'] + channel_names # label 0: na. Let channel start from 1.

        # Process annotations to be frame-level (75Hz)
        annotations = sample_meta['annotations']
        n_frames = self.config.label_frame_rate * self.config.segment_length
        frame_label = torch.zeros(size=(n_frames,), dtype=torch.int64)
        for annot_entry in annotations:
            start_frame = int(timecode_to_seconds(annot_entry['start']) * self.config.label_frame_rate)
            end_frame = int(timecode_to_seconds(annot_entry['end']) * self.config.label_frame_rate)
            lead_inst_ch_name = annot_entry['lead']
            lead_inst_ch_id = channel_names.index(lead_inst_ch_name)
            frame_label[start_frame:end_frame] = lead_inst_ch_id


        ret = {
            'audio_stack': audio_stack, # tensor of shape [n_inst, n_samples=120320]
            'inst_ids': inst_ids, # tensor of shape [n_inst]
            'label': frame_label, # tensor of shape [n_frames]
        }

        return ret

    def collate_fn(self, batch):
        '''
        Collate function for MJNDatasetCh

        if a batch contains different instruments, inst dim will be padded with zeros to the audio with most instruments
        
        Output format: 
        {
            audio: [bs, n_inst, n_samples=120320]
            inst_ids: [bs, n_inst]  value: one of instrument id
            label: [bs, n_frames]   valud: [0 ~ n_inst], 0: silent; 1 ~ n_inst: channel
        }
            
        '''
        audio = []
        inst_ids = []
        label = []
        for sample in batch:
            audio.append(sample['audio_stack'])
            inst_ids.append(sample['inst_ids'])
            label.append(sample['label'])

        # Ensure all audio have the same number of samples
        n_samples = self.config.segment_length * self.config.tgt_sr + int(self.config.tgt_sr / self.config.label_frame_rate)
        # audio in shape [bs, n_inst, n_samples=120320]
        # Enumerate all audio
        for i, audio_stack in enumerate(audio): 
            # If audio is shorter, pad with zeros
            if audio_stack.size(1) < n_samples:
                pad = torch.zeros(size=(audio_stack.size(0), n_samples - audio_stack.size(1)))
                audio[i] = torch.cat([audio_stack, pad], dim=1)
            # If audio is longer, truncate
            elif audio_stack.size(1) > n_samples:
                audio[i] = audio_stack[:, :n_samples]

        # Pad audio to the same number of instruments
        n_channels = [audio_i.size(0) for audio_i in audio]
        max_n_channels = max(n_channels)
        # ch_pad_masks = []
        for i, audio_stack in enumerate(audio):
            if audio_stack.size(0) < max_n_channels:
                pad = torch.zeros(size=(max_n_channels - audio_stack.size(0), audio_stack.size(1)))
                audio[i] = torch.cat([audio_stack, pad], dim=0)
        audio = torch.stack(audio, dim=0) # [bs, max_n_inst, n_samples=120320]

        # Pad inst_ids to the same number of channels
        for i, inst_ids_i in enumerate(inst_ids):
            if inst_ids_i.size(0) < max_n_channels:
                pad = torch.zeros(size=(max_n_channels - inst_ids_i.size(0),), dtype=torch.int64)
                inst_ids[i] = torch.cat([inst_ids_i, pad], dim=0)
        inst_ids = torch.stack(inst_ids, dim=0) # [bs, max_n_insts]

        # Collate label
        label = torch.stack(label, dim=0) # [bs, n_frames]

        ret = {
            'audio': audio,   # [bs, max_n_inst, n_samples=120320]
            # 'ch_pad_masks': ch_pad_masks,
            'inst_ids': inst_ids,  # [bs, max_n_insts]
            'label': label,   # [bs, n_frames]
        }

        return ret


class MJNDatasetSegment(MJNDataset):
    '''
    Dataset for segment-level guitar solo binary classification
    '''

    def __getitem__(self, idx):
        '''
        Return 
        - a 5-s clip of multi-channel audios. Each channel is a different instrument.
        - Instrument of each channel
        - Label of the clip
        '''
        sample_meta = self.meta[idx]
        audio_dirpath = jpath(self.root, sample_meta['seg_audio_dir'])
        # audio_fns = ls(audio_dirpath)
        audio_fn = 'mix.mp3'

        # Read audios, convert to mono, stack together
        # audio_fps = [jpath(audio_dirpath, audio_fn) for audio_fn in audio_fns]
        # audios = [torchaudio.load(audio_fp)[0] for audio_fp in audio_fps]

        audio_fp = jpath(audio_dirpath, audio_fn)
        audio = torchaudio.load(audio_fp)[0]

        # Resample to target sampling rate
        resampler = torchaudio.transforms.Resample(orig_freq=self.config.src_sr, new_freq=self.tgt_sr)
        # audios = [resampler(audio) for audio in audios]
        audio = resampler(audio)

        # Right side zero pad 320 samples for each audio
        audio = torch.nn.functional.pad(audio, (0, 320))
        audio = convert_waveform_to_mono(audio) # each element: [1, n_samples=120320]

        # Process annotations to be segment-level (75Hz)
        annotations = sample_meta['annotations']
        guitar_dur = 0
        for annot_entry in annotations:
            if 'guitar' in annot_entry['lead']:
                start_frame = timecode_to_seconds(annot_entry['start'])
                end_frame = timecode_to_seconds(annot_entry['end'])
                guitar_dur += end_frame - start_frame
        if guitar_dur > 2.5:
            label = 1
        else:
            label = 0

        ret = {
            'audio_stack': audio, # tensor of shape [n_inst, n_samples=120320]
            'inst_ids': None, # tensor of shape [n_inst]
            'label': label, # tensor of shape [n_frames]
        }

        return ret
    
    def collate_fn(self, batch):
        '''
        Collate function for MJNDataset

        if a batch contains different instruments, inst dim will be padded with zeros to the audio with most instruments
        
        Output format: 
        {
            audio: [bs, n_samples=120320]
            inst_ids: None
            label: [bs, ]   valud: [0 ~ n_inst], 0: silent; 1 ~ n_inst: channel
        }
            
        '''
        audio = []
        inst_ids = []
        label = []
        for sample in batch:
            audio.append(sample['audio_stack'])
            inst_ids.append(sample['inst_ids'])
            label.append(sample['label'])

        # Ensure all audio have the same number of samples
        n_samples = self.config.segment_length * self.config.tgt_sr + int(self.config.tgt_sr / self.config.label_frame_rate)
        # audio in shape [bs, n_inst, n_samples=120320]
        # Enumerate all audio
        for i, audio_stack in enumerate(audio): 
            # If audio is shorter, pad with zeros
            if audio_stack.size(1) < n_samples:
                pad = torch.zeros(size=(audio_stack.size(0), n_samples - audio_stack.size(1)))
                audio[i] = torch.cat([audio_stack, pad], dim=1)
            # If audio is longer, truncate
            elif audio_stack.size(1) > n_samples:
                audio[i] = audio_stack[:, :n_samples]

        audio = torch.stack(audio, dim=0) # [bs, n_samples=120320]
        label = torch.tensor(label) # [bs, ]

        ret = {
            'audio': audio,   # [bs, max_n_inst, n_samples=120320]
            'inst_ids': None,  # [bs, max_n_insts]
            'label': label,   # [bs, ]
        }

        return ret


class CrossDatasetCh(MJNDatasetCh):
    '''
    Read both MJN and MedleyDB dataset at the same time, for training channel classification model
    '''
    
    def __init__(self, dataset_config, split, tgt_sr=24000):
        super().__init__(dataset_config, split, tgt_sr)

        # Read two datasets
        mjn_meta = read_json(jpath(dataset_config.mjn_root, 'metadata.json'))
        medley_meta = read_json(jpath(dataset_config.medleydb_root, 'metadata.json'))

        # Replace the audio dir to the full path
        for k in mjn_meta:
            mjn_meta[k]['seg_audio_dir'] = jpath(dataset_config.mjn_root, mjn_meta[k]['seg_audio_dir'])
        for k in medley_meta:
            medley_meta[k]['seg_audio_dir'] = jpath(dataset_config.medleydb_root, medley_meta[k]['seg_audio_dir'])        

        # Merge the metadata
        self.meta = {**mjn_meta, **medley_meta}

        # Filter by split
        self.meta = {k: v for k, v in self.meta.items() if v['split'] == self.split}

        # Re-index self.meta with key start from 0, original key as a new field 'id'
        for i, v in self.meta.items():
            v['id'] = i
        self.meta = {i: v for i, v in enumerate(self.meta.values())}

    def __getitem__(self, idx):
        '''
        Return 
        - audios: a 5-s clip of multi-channel audios. Each channel is a different instrument.
        - inst ids: Instrument of each channel
        - labels: Label of the clip. 0: silent; 1 ~ n_inst: channel
        '''
        sample_meta = self.meta[idx]

        audio_dirpath = sample_meta['seg_audio_dir']
        audio_fns = ls(audio_dirpath)

        # Dataset specific setting
        if 'mix.mp3' in audio_fns: # MJN dataset
            mix_fn = 'mix.mp3'
        else:   # MedleyDB dataset
            mix_fn = '0#mix.mp3'
        
        audio_fns.remove(mix_fn)
        if self.split == 'train':
            # Delete the suffix of the audio filenames
            ch_names = [audio_fn.split('.')[0] for audio_fn in audio_fns]

            # Channel permutation augmentation
            if self.config.get('ch_permute_aug', False):
                random.shuffle(ch_names)

            # Recover the audio filenames
            audio_fns = [ch_name + '.mp3' for ch_name in ch_names]
        # Ensure the mix is the first channel
        audio_fns.insert(0, mix_fn)

        # Read audios, convert to mono, stack together
        audio_fps = [jpath(audio_dirpath, audio_fn) for audio_fn in audio_fns]
        audios = [torchaudio.load(audio_fp)[0] for audio_fp in audio_fps]

        # Resample to target sampling rate
        resampler = torchaudio.transforms.Resample(orig_freq=self.config.src_sr, new_freq=self.tgt_sr)
        audios = [resampler(audio) for audio in audios]

        # Right side zero pad 320 samples for each audio
        audios = [torch.nn.functional.pad(audio, (0, 320)) for audio in audios]
        audios = [convert_waveform_to_mono(audio) for audio in audios]
        audio_stack = torch.cat(audios, dim=0) # [n_inst, n_samples=120320]

        # Get channel names of each channel
        channel_names = [audio_fn.split('.')[0] for audio_fn in audio_fns] # normalized stem names
        if self.config.get('cross_dataset_tokenizer', False):
            inst_names = [inst_utils_cd.from_stem_name_get_inst_name(channel_name) for channel_name in channel_names]
        else:
            inst_names = [inst_utils.from_stem_name_get_inst_name(channel_name) for channel_name in channel_names]
        inst_ids = torch.tensor([self.inst_tokenizer.convert_inst_to_id(inst_name) for inst_name in inst_names], dtype=torch.int64)
        channel_names = ['na'] + channel_names # label 0: na. Let channel start from 1.

        # Process annotations to be frame-level (75Hz)
        annotations = sample_meta['annotations']
        n_frames = self.config.label_frame_rate * self.config.segment_length
        frame_label = torch.zeros(size=(n_frames,), dtype=torch.int64)
        for annot_entry in annotations:
            start_frame = int(timecode_to_seconds(annot_entry['start']) * self.config.label_frame_rate)
            end_frame = int(timecode_to_seconds(annot_entry['end']) * self.config.label_frame_rate)
            lead_inst_ch_name = annot_entry['lead']
            lead_inst_ch_id = channel_names.index(lead_inst_ch_name)
            frame_label[start_frame:end_frame] = lead_inst_ch_id


        ret = {
            'audio_stack': audio_stack, # tensor of shape [n_inst, n_samples=120320]
            'inst_ids': inst_ids, # tensor of shape [n_inst]
            'label': frame_label, # tensor of shape [n_frames]
        }

        return ret
