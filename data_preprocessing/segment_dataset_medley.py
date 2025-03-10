'''
Create segment-level dataset for MedleyDB v1 dataset
From the stem audios and lead instrument annotation

Author: Longshen Ou, 2024/07/19
'''

import os
import sys

sys.path.append('.')
sys.path.append('..')

import json
import random
import datetime
import torchaudio
import pandas as pd
import inst_utils_cross_dataset as inst_utils_cd
from utils import *
from tqdm import tqdm
from typing import List, Dict


def main():
    pass


def procedures():
    check_annotation_legality()

    create_metadata_for_segmented_dataset() # Prepare segment-level annotation, generate metadata_seg.json
    segment_audio()                         # Segment the audio according to the timecode in the prepared annotation
    split_dataset()                         # Split the dataset into train, valid, test, generate metadata_splitted.json
    refine_seg_meta()                       # Clean the metadata for the segmented dataset, generate metadata.json
    check_inst_define()                     # Check if the instrument definition is good for all stem name


def check_inst_define():
    seg_dataset_root = '/home/longshen/data/datasets/MedleyDB/v1_segmented'
    meta_fp = jpath(seg_dataset_root, 'metadata.json')
    meta = read_json(meta_fp)
    all_insts = set()
    n_ch = {}
    for seg_id in meta:
        seg = meta[seg_id]
        audio_dir = seg['seg_audio_dir']
        audio_dir = jpath(seg_dataset_root, audio_dir)
        audio_fns = ls(audio_dir)
        update_dict_cnt(n_ch, len(audio_fns))
        for audio_fn in audio_fns:
            stem_name = audio_fn.split('.')[0]
            inst = inst_utils_cd.from_stem_name_get_inst_name(stem_name)
            all_insts.add(inst)

    # Sort n_ch by key
    n_ch = dict(sorted(n_ch.items(), key=lambda item: item[0], reverse=True))
    save_dir = '/home/longshen/work/SoloDetection/misc/medleyDB'
    save_json(n_ch, jpath(save_dir, 'n_ch.json'))



def split_dataset():
    seg_dataset_root = '/Users/longshen/Code/Datasets/MedleyDB/v1_segmented'
    meta_fp = jpath(seg_dataset_root, 'metadata_seg.json')
    meta = read_json(meta_fp)
    new_meta_fp = jpath(seg_dataset_root, 'metadata_splitted.json')
    
    # Split different songs into train, validation, test
    # One song only appears in one split
    song_names = list(meta.keys())
    n_songs = len(song_names)

    # Choose validation and testing split
    valid_ratio = 0.15
    test_ratio = 0.15
    n_valid = int(n_songs * valid_ratio)
    n_test = int(n_songs * test_ratio)
    
    # Split
    all_indices = list(range(n_songs))
    random.shuffle(all_indices)
    valid_indices = all_indices[:n_valid]
    test_indices = all_indices[n_valid:n_valid+n_test]
    train_indices = all_indices[n_valid+n_test:]

    valid_song_names = [song_names[i] for i in valid_indices]
    test_song_names = [song_names[i] for i in test_indices]
    train_song_names = [song_names[i] for i in train_indices]

    split_indices = {}
    for song_name in meta:
        if song_name in train_song_names:
            split_indices[song_name] = 'train'
        elif song_name in valid_song_names:
            split_indices[song_name] = 'valid'
        elif song_name in test_song_names:
            split_indices[song_name] = 'test'
        else:
            raise ValueError('Unknown split for {}'.format(song_name))
        
    for song_name in meta:
        for i, seg in enumerate(meta[song_name]):
            meta[song_name][i]['split'] = split_indices[song_name]

    save_json(meta, new_meta_fp)


def refine_seg_meta():
    '''
    Refine the segment-level metadata

    Discard
        - Song-segment hierarchy (flattened)
    Convert
        - Annotation timing to relative timing
    Add:
        - Path to audio dir
        - Original start and end time
    '''
    # Prepare paths
    segmented_dir = '/Users/longshen/Code/Datasets/MedleyDB/v1_segmented'
    seg_meta_fp = jpath(segmented_dir, 'metadata_splitted.json')
    audio_seg_root = jpath(segmented_dir, 'data')
    seg_meta = read_json(seg_meta_fp)
    meta_fp = jpath(segmented_dir, 'metadata.json')
    meta = {}

    for song_name in tqdm(seg_meta):
        segs = seg_meta[song_name]
        for i, seg in enumerate(segs):
            segment_id = str(i)
            raw_annotations = seg['annotations']
            ori_start = seg['segment_start']
            ori_end = seg['segment_end']
            annotations = []
            ori_start_timedelta = timecode_to_timedelta(ori_start)
            ori_end_timedelta = timecode_to_timedelta(ori_end)
            for raw_annot in raw_annotations:
                annot_start_delta = timecode_to_timedelta(raw_annot['lead_start'])
                if ori_start_timedelta > annot_start_delta:
                    annot_rel_start_time = datetime.timedelta(seconds=0.)
                else:
                    annot_rel_start_time = annot_start_delta - ori_start_timedelta
                annot_end_delta = timecode_to_timedelta(raw_annot['lead_end'])
                if annot_end_delta <= ori_end_timedelta:
                    annot_rel_end_time = annot_end_delta - ori_start_timedelta
                else:
                    annot_rel_end_time = ori_end_timedelta - ori_start_timedelta
                annotations.append({
                    'start': timedelta_to_timecode(annot_rel_start_time),
                    'end': timedelta_to_timecode(annot_rel_end_time),
                    'lead': raw_annot['lead_inst']
                })

            audio_seg_dir = jpath('data', str(song_name), str(segment_id))

            entry = {
                'seg_audio_dir': audio_seg_dir,
                'annotations': annotations,
                'original_start': ori_start,
                'original_end': ori_end,
                'split': seg['split'],
            }

            meta['{}-{}'.format(song_name, i)] = entry
            
            
    save_json(meta, meta_fp)


def segment_audio():
    '''
    Do the segmentation according to the metadata_seg.json

    NOTE: segmented audio are saved in mp3 format, with filename as the stem name
    '''
    # Prepare paths
    dataset_root = '/Users/longshen/Code/Datasets/MedleyDB/v1_mp3/Audio'
    segmented_dir = '/Users/longshen/Code/Datasets/MedleyDB/v1_segmented'
    seg_meta_fp = jpath(segmented_dir, 'metadata_seg.json')

    # Create dir for audio segments
    audio_seg_dir = jpath(segmented_dir, 'data')
    create_if_not_exist(audio_seg_dir)

    # Create dir for performance
    seg_meta = read_json(seg_meta_fp)
    for song_name in tqdm(seg_meta):
        seg_info = seg_meta[song_name] # A list of segment entries

        song_audio_save_dir = jpath(audio_seg_dir, song_name)
        create_if_not_exist(song_audio_save_dir)

        # Read the song metadata
        song_meta_fp = jpath(dataset_root, song_name, '{}_METADATA.yaml'.format(song_name))
        if not os.path.exists(song_meta_fp):
            print('No metadata file for {}'.format(song_name))
            continue
        song_meta = read_yaml(song_meta_fp)
        stems_info = song_meta['stems']
        
        track_audios = {}
        for stem_s_name, stem_info in stems_info.items():
            stem_id = int(stem_s_name[1:])
            inst_name = stem_info['instrument']
            filename = stem_info['filename']

            # Read the stem audio
            filename = filename.replace('.wav', '.mp3')
            audio_fp = jpath(dataset_root, song_name, '{}_STEMS'.format(song_name), filename)
            stem_audio, ori_sr = torchaudio.load(audio_fp)
            stem_audio = normalize_waveform(stem_audio)

            inst_name_normalized = inst_name.replace('/', '_').replace(' ', '_')
            stem_name = '{}#{}'.format(stem_id, inst_name_normalized)

            track_audios[stem_name] = stem_audio
        
        # Read the mixture audio
        mix_audio_fp = jpath(dataset_root, song_name, '{}_MIX.mp3'.format(song_name))
        assert os.path.exists(mix_audio_fp), 'Mixture audio not found: {}'.format(mix_audio_fp)
        mix_audio, ori_sr = torchaudio.load(mix_audio_fp)
        mix_audio = normalize_waveform(mix_audio)
        mixture_stem_name = '0#mix'
        track_audios[mixture_stem_name] = mix_audio
            
        # Iterate over segment-level info of the song
        for i, seg in enumerate(tqdm(seg_info)):
            seg_start = seg['segment_start']
            seg_end = seg['segment_end']
            
            # Create segment dir
            segment_name = i
            segment_dir = jpath(song_audio_save_dir, '{}'.format(segment_name))
            create_if_not_exist(segment_dir)

            for stem_name in track_audios:
                track_audio_seg_fp = jpath(segment_dir, '{}.mp3'.format(stem_name))
                track_audio = track_audios[stem_name]
                save_segment_from_mp3(track_audio, ori_sr, track_audio_seg_fp, seg_start, seg_end) # TODO: progress, code to segment the audio
        

def save_segment_from_mp3(waveform, sr, output_file, start_time_str, end_time_str, format='mp3'):
    # Convert times to sample indices
    start_sample = time_str_to_samples(start_time_str, sr)
    end_sample = time_str_to_samples(end_time_str, sr)

    # Extract the segment
    segment = waveform[:, start_sample:end_sample]

    # Save the extracted segment
    torchaudio.save(output_file, segment, sr, format=format)


def time_str_to_samples(time_str, sample_rate):
    """Convert a time string formatted as 'MM:SS.MS' to sample index."""
    minutes, seconds = time_str.split(':')
    total_seconds = int(minutes) * 60 + float(seconds)
    return int(total_seconds * sample_rate)


def check_annotation_legality():
    '''
    Check if the annotation is legal
    '''

    # Prepare paths
    dataset_root = '/Users/longshen/Code/Datasets/MedleyDB/v1_mp3/Audio'
    
    # Get annotation filenames
    song_dirnames = ls(dataset_root)

    # Check legality of annotation
    pbar = tqdm(song_dirnames)
    for song_dirname in pbar:
        pbar.set_description(song_dirname)

        song_dirpath = jpath(dataset_root, song_dirname)
        annot_fp = jpath(song_dirpath, 'Markers.csv')

        if not os.path.exists(annot_fp):
            print('No annotation file for {}'.format(song_dirname))
            continue

        annot = read_csv_to_json(annot_fp)

        # Check if the annotation is legal
        for entry in annot:
            assert entry['Start'] is not None, 'Start time missing in entry {} from {}'.format(entry, song_dirname)
            assert entry['Duration'] is not None, 'Duration missing in entry {} from {}'.format(entry, song_dirname)
            dur = timecode_to_millisecond(entry['Duration'])
            assert dur >= 0, 'Negative or zero duration: {}'.format(entry)
            description = entry['Description']
            assert isinstance(description, int), 'Illegal instrument description in entry {} from {}'.format(entry, song_dirname)

    print('Complete!')


def create_metadata_for_segmented_dataset():
    '''
    Create metadata.json for the dataset after segmentation.

    NOTE: instrument name is normalized to stem name

    Include below information:
    {
        "XXXJingleBell": [ # Song name
            {
                "segment_start": "00:10.381",
                "segment_end": "00:15.381",
                "annotations": [
                    {
                        "lead_start": "00:00.000",
                        "lead_end": "00:20.381",
                        "lead_inst": "e-guitar#3"               # MedleyDB stem name, in the format of "[inst name]#[channel id]"
                    }
                ]
            },
            {
        ...
    '''
    # Prepare paths
    dataset_root = '/Users/longshen/Code/Datasets/MedleyDB/v1_mp3/Audio'
    segmented_dir = '/Users/longshen/Code/Datasets/MedleyDB/v1_segmented'
    seg_meta_fp = jpath(segmented_dir, 'metadata_seg.json')
    
    res = {}

    # Iterative over all performances' annotation file
    song_dirnames = ls(dataset_root)
    pbar = tqdm(song_dirnames)
    for song_dirname in pbar: # for all performance
        pbar.set_description(song_dirname)

        song_dirpath = jpath(dataset_root, song_dirname)
        annot_fp = jpath(song_dirpath, 'Markers.csv')

        if not os.path.exists(annot_fp):
            print('No annotation file for {}'.format(song_dirname))
            continue
        annot = read_csv_to_json(annot_fp)

        # Read the original song-level metadata 
        meta_fp = jpath(song_dirpath, '{}_METADATA.yaml'.format(song_dirname))
        assert os.path.exists(meta_fp), 'Metadata file not found: {}'.format(meta_fp)
        meta = read_yaml(meta_fp)

        # Obtain channel id to stem name mapping
        ch_id_to_stem_name = {}
        all_stem_info = meta['stems']
        for stem_name, stem_info in all_stem_info.items():
            stem_id = int(stem_name[1:])
            inst_name = stem_info['instrument']
            inst_name_normalized = inst_name.replace('/', '_').replace(' ', '_')
            stem_name = '{}#{}'.format(stem_id, inst_name_normalized)
            ch_id_to_stem_name[stem_id] = stem_name

        # Normalize the instrument name
        for i, entry in enumerate(annot):
            lead_ch_id = entry['Description']

            lead_stem_name = ch_id_to_stem_name[lead_ch_id]
            entry['Description'] = lead_stem_name

        relevant_segments = get_relevant_segments(annot)
        
        all_segmented_data = segment_annotations_based_on_relevance(annot, relevant_segments)

        res[song_dirname] = all_segmented_data


    create_if_not_exist(segmented_dir)
    save_json(res, seg_meta_fp)


def get_relevant_segments(annot: List[Dict], buffer = 2.0):
    '''
    Assume we have at least 10 s before and after the recording
    '''
    import datetime
    buffer = datetime.timedelta(seconds=buffer)

    relevant_segments = []
    last_end_time = datetime.timedelta(seconds=0)
    for i, entry in enumerate(annot):
        type = entry['Description']
        # if type == 'na' or type == 'NA' or type is None:
        #     a = 1
        if type == None: # If a non-song part comes
            continue

        start_time = timecode_to_timedelta(entry['Start']) # assumption: all performance start with NA, end with NA
        duration = timecode_to_timedelta(entry['Duration'])
        end_time = start_time + duration

        start_buffered = max(datetime.timedelta(seconds=0), start_time- buffer)
        end_buffered = end_time + buffer

        # Merge overlapping or close segments
        if start_buffered <= last_end_time and len(relevant_segments) > 0:
            # Extend the current segment if overlapping or directly adjacent within buffer
            relevant_segments[-1] = (relevant_segments[-1][0], max(relevant_segments[-1][1], end_buffered))
        else:
            # Start a new segment
            relevant_segments.append((start_buffered, end_buffered))
        
        last_end_time = end_buffered
    
    return relevant_segments
        


def segment_annotations_based_on_relevance(annot, relevant_segments, segment_length=5.0, overlap=2.5):
    """
    Generate segments with metadata from specified relevant audio segments, handling list of dictionaries annotations.

    Parameters:
    annot (list): List of dictionaries containing annotations with keys like 'Start', 'Duration', etc.
    relevant_segments (list): List of tuples, each containing (start_time, end_time) of relevant audio segments.
    segment_length (float): Length of each segment in seconds.
    overlap (float): Overlap between consecutive segments in seconds.

    Returns:
    list: List of dictionaries, each containing metadata for a segment.
    """
    import datetime
    all_segmented_data = []

    segment_length = datetime.timedelta(seconds=segment_length)
    overlap = datetime.timedelta(seconds=overlap)

    # Convert start times and durations in annotations to seconds
    for i, annotation in enumerate(annot):
        annotation['Start'] = timecode_to_timedelta(annotation['Start']) # Inplace modification
        annotation['End'] = annotation['Start'] + timecode_to_timedelta(annotation['Duration']) # Inplace modification

    # Process each relevant segment
    # pbar = tqdm(relevant_segments)
    for start_relevant, end_relevant in relevant_segments:
        
        current_time = start_relevant
        while current_time + segment_length <= end_relevant:
            segment_end = current_time + segment_length
            
            # Filter annotations that are active during the current segment
            segment_annotations = [
                ann for ann in annot
                if ann['Start'] < segment_end and (ann['End']) > current_time
            ]

            # Prepare metadata for the segment
            segment_data = {
                'segment_start': timedelta_to_timecode(current_time),
                'segment_end': timedelta_to_timecode(segment_end),
                'annotations': []
            }

            # Collect all annotations within the segment
            for ann in segment_annotations:
                segment_data['annotations'].append({
                    'lead_start': timedelta_to_timecode(ann['Start']),
                    'lead_end': timedelta_to_timecode(ann['End']),
                    'lead_inst': ann.get('Description') or 'na'  # Using 'Name' if 'Description' is None
                })

            all_segmented_data.append(segment_data)
            current_time += segment_length - overlap  # Apply overlap
    
    return all_segmented_data


def read_csv_to_json(excel_fp):
    '''
    Read excel file with pandas to DataFrame,
    Then convert to json.
    Format: [{column -> value}, … , {column -> value}]
    '''
    meta_raw_audio_dir = pd.read_csv(excel_fp, sep='\t') # class: pd.DataFrame
    meta_raw_audio_dir = meta_raw_audio_dir.fillna('na')
    t = meta_raw_audio_dir.to_json(orient='records')
    ret = json.loads(t)
    return ret


if __name__ == '__main__':
    main()