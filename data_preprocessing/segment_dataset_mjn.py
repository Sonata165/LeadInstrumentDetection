'''
Create segment-level dataset dataset for MJN dataset
From the stem audios and lead instrument annotation

Author: Longshen Ou, 2024/06/05
'''

import os
import sys

sys.path.append('.')
sys.path.append('..')

import json
import datetime
import torchaudio
import pandas as pd
from utils import *
from tqdm import tqdm
from typing import List, Dict
from inst_utils import from_audio_name_get_stem_name, from_inst_abbr_get_stem_name, inst_abbr_to_stem_name, stem_audio_fn_to_stem_name, stem_inst_map


def main():
    pass


def procedures():
    check_annotation_legality()
    check_stem_audio_legality()

    create_metadata_for_segmented_dataset() # Prepare segment-level annotation, generate metadata_seg.json
    segment_audio()                         # Segment the audio according to the timecode in the prepared annotation
    split_dataset()                         # Split the dataset into train, valid, test, generate metadata_splitted.json
    refine_seg_meta()                       # Clean the metadata for the segmented dataset, generate metadata.json


def remove_na_from_new_meta():
    '''
    This function does not work, because once additional na segments are introduced,
    the segment start time in the clean meta is not consistent with the original start time in the new meta.
    '''
    clean_meta_fp = '/Users/longshen/Code/Datasets/MJN/segmented/5perfs/metadata_seg_debug.json'
    meta_fp = '/Users/longshen/Code/Datasets/MJN/segmented/5perfs/metadata_bak.json'
    save_fp = '/Users/longshen/Code/Datasets/MJN/segmented/5perfs/metadata.json'

    meta = read_json(meta_fp)
    clean_meta = read_json(clean_meta_fp)
    res = {}

    # Get the segment start time in clean meta
    seg_start_time_dict = {}
    for perf_id, seg_list in clean_meta.items():
        start_times = set()
        for i, seg in enumerate(seg_list):
            start_times.add(seg['segment_start'])
        seg_start_time_dict[perf_id] = start_times

    for perf_seg_id, entry in tqdm(meta.items()):
        t = perf_seg_id.split('-')
        perf_id = '-'.join(t[:-1])
        seg_id = int(t[-1])

        if entry['original_start'] not in seg_start_time_dict[perf_id]:
            continue
        else:
            res[perf_seg_id] = entry

    save_json(res, save_fp)



def split_dataset():
    seg_dataset_root = '/Users/longshen/Code/Datasets/MJN/segmented/5perfs'
    meta_fp = jpath(seg_dataset_root, 'metadata_seg.json')
    meta = read_json(meta_fp)
    new_meta_fp = jpath(seg_dataset_root, 'metadata_splitted.json')
    
    # Split different performances into train, test
    # One performance only appears in one split
    pfm_ids = meta.keys()
    n_pfm = len(pfm_ids)

    # Choose validation and testing split
    valid_pfm_ids = ['2-5', '14-5', '15-1']
    test_pfm_ids = ['2-2', '15-3']
    train_pfm_ids = [pfm_id for pfm_id in pfm_ids if pfm_id not in valid_pfm_ids and pfm_id not in test_pfm_ids]

    split_indices = {}
    for pfm_id in meta:
        if pfm_id in train_pfm_ids:
            split_indices[pfm_id] = 'train'
        elif pfm_id in valid_pfm_ids:
            split_indices[pfm_id] = 'valid'
        else:
            split_indices[pfm_id] = 'test'
        
    for pfm_id in meta:
        for i, seg in enumerate(meta[pfm_id]):
            meta[pfm_id][i]['split'] = split_indices[pfm_id]

    save_json(meta, new_meta_fp)


def refine_seg_meta():
    '''
    Refine the segment-level metadata

    Discard
        - Event - Performance hierarchy (flattened)
    Convert
        - Annotation timing to relative timing
    Add:
        - Path to audio dir
        - Types of instruments
        - Relevative annotion start and end timecode
    '''
    # Prepare paths
    dataset_root = '/Users/longshen/Code/Datasets/MJN/Preliminary'
    segmented_dir = '/Users/longshen/Code/Datasets/MJN/segmented/5perfs'

    seg_meta_fp = jpath(segmented_dir, 'metadata_splitted.json')
    audio_seg_root = jpath(segmented_dir, 'data')
    seg_meta = read_json(seg_meta_fp)
    meta_fp = jpath(segmented_dir, 'metadata.json')
    meta = {}

    for performance_id in tqdm(seg_meta):
        segs = seg_meta[performance_id]
        for i, seg in enumerate(segs):
            segment_id = str(i)
            raw_annotations = seg['annotations']
            ori_start = seg['segment_start']
            ori_end = seg['segment_end']
            annotations = []
            ori_start_delta = timecode_to_timedelta(ori_start)
            ori_end_delta = timecode_to_timedelta(ori_end)
            for raw_annot in raw_annotations:
                annot_start_delta = timecode_to_timedelta(raw_annot['lead_start'])
                if ori_start_delta > annot_start_delta:
                    annot_rel_start_time = datetime.timedelta(seconds=0.)
                else:
                    annot_rel_start_time = annot_start_delta - ori_start_delta
                annot_end_delta = timecode_to_timedelta(raw_annot['lead_end'])
                if annot_end_delta <= ori_end_delta:
                    annot_rel_end_time = annot_end_delta - ori_start_delta
                else:
                    annot_rel_end_time = ori_end_delta - ori_start_delta
                annotations.append({
                    'start': timedelta_to_timecode(annot_rel_start_time),
                    'end': timedelta_to_timecode(annot_rel_end_time),
                    'lead': raw_annot['lead_inst']
                })

            audio_seg_dir = jpath('data', str(performance_id), str(segment_id))

            entry = {
                'seg_audio_dir': audio_seg_dir,
                'annotations': annotations,
                'original_start': ori_start,
                'original_end': ori_end,
                'split': seg['split'],
            }

            meta['{}-{}'.format(performance_id, i)] = entry
            
            
    save_json(meta, meta_fp)

def segment_audio():
    '''
    Do the segmentation according to the metadata_seg.json

    NOTE: segmented audio are saved in mp3 format, with filename as the stem name
    '''
    # Prepare paths
    dataset_root = '/Users/longshen/Code/Datasets/MJN/Preliminary'
    segmented_dir = jpath('/Users/longshen/Code/Datasets/MJN', 'segmented', '5perfs')
    seg_meta_fp = jpath(segmented_dir, 'metadata_seg.json')
    annotation_dir = jpath(dataset_root, 'annotations')
    stem_audio_root = jpath(dataset_root, 'stem_audio')
    target_sr = 48000

    # Create dir for audio segments
    audio_seg_dir = jpath(segmented_dir, 'data')
    create_if_not_exist(audio_seg_dir)

    # Create dir for performance
    seg_meta = read_json(seg_meta_fp)
    for perf_id in tqdm(seg_meta):
        seg_info = seg_meta[perf_id] # A list of segment entries

        perf_dir = jpath(audio_seg_dir, perf_id)
        create_if_not_exist(perf_dir)

        # Read stem audios
        stem_audio_dir = jpath(stem_audio_root, perf_id)
        os.path.exists(stem_audio_dir)
        track_fns = ls(stem_audio_dir)
        track_audios = {}
        for track_fn in track_fns:
            inst = track_fn.split('.')[0].lower()
            track_fp = jpath(stem_audio_dir, track_fn)
            track_audio, _ = torchaudio.load(track_fp)
            track_audios[inst] = track_audio

        # Iterate over segment-level info of the song
        for i, seg in enumerate(tqdm(seg_info)):
            seg_start = seg['segment_start']
            seg_end = seg['segment_end']
            
            # Create segment dir
            segment_name = i
            segment_dir = jpath(perf_dir, '{}'.format(segment_name))
            create_if_not_exist(segment_dir)

            for inst in track_audios:
                track_audio_seg_fp = jpath(segment_dir, '{}.mp3'.format(from_audio_name_get_stem_name(inst)))
                track_audio = track_audios[inst]
                save_segment_from_mp3(track_audio, target_sr, track_audio_seg_fp, seg_start, seg_end) # TODO: progress, code to segment the audio
                

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
    dataset_root = '/Users/longshen/Code/Datasets/MJN/Preliminary'
    segmented_dir = jpath(dataset_root, 'segmented_dataset_4perfs')
    seg_meta_fp = jpath(segmented_dir, 'metadata_seg.json')
    annotation_dir = jpath(dataset_root, 'annotations')
    
    # Get annotation filenames
    annot_fns = [fn for fn in ls(annotation_dir) if not fn.startswith('.')]
    perform_ids = [i.strip().split('.')[0] for i in annot_fns]

    res = {}

    # Check legality of annotation
    pbar = tqdm(annot_fns)
    for annot_fn in pbar:
        pbar.set_description(annot_fn)
        perform_id, band_id = annot_fn.split('-')
        annot_fp = jpath(annotation_dir, annot_fn)
        annot = read_csv_to_json(annot_fp)

        # Check if the annotation is legal
        for entry in annot:
            assert entry['Start'] is not None, 'Start time missing in entry {} from {}'.format(entry, annot_fn)
            assert entry['Duration'] is not None, 'Duration missing: {}'.format(annot_fn)
            dur = timecode_to_millisecond(entry['Duration'])
            assert dur >= 0, 'Negative or zero duration: {}'.format(annot_fn)
            description = entry['Description']
            assert description is not None, 'Description missing in entry {} from {}'.format(entry, annot_fn)
            assert description.lower() in inst_abbr_to_stem_name, 'Illegal instrument description in entry {} from {}'.format(entry, annot_fn)

    print('Complete!')


def check_stem_audio_legality():
    '''
    Check if the stem audio is legal 
    '''

    dataset_root = '/Users/longshen/Code/Datasets/MJN/Preliminary'
    stem_audio_root = jpath(dataset_root, 'stem_audio')
    audio_dns = ls(stem_audio_root)
    pbar = tqdm(audio_dns)
    for audio_dn in audio_dns:
        pbar.set_description(audio_dn)
        
        perf_id, band_id = audio_dn.split('-')
        band_dp = jpath(stem_audio_root, audio_dn)

        audio_fns = ls(band_dp)
        for audio_fn in audio_fns:
            audio_fn_name = audio_fn.split('.')[0].lower()
            assert audio_fn_name in stem_audio_fn_to_stem_name, 'No stem name for audio with name: {}'.format(audio_fn_name)
            stem_name = stem_audio_fn_to_stem_name[audio_fn_name]
            assert stem_name in stem_inst_map, 'No inst def for stem name: {}'.format(stem_name)
    print('Complete!')



def create_metadata_for_segmented_dataset():
    '''
    Create metadata.json for the dataset after segmentation.

    NOTE: instrument name is normalized to stem name

    Include below information:
    {
        "2-1": [
            {
                "segment_start": "00:10.381",
                "segment_end": "00:15.381",
                "annotations": [
                    {
                        "lead_start": "00:00.000",
                        "lead_end": "00:20.381",
                        "lead_inst": "na"               # stem name
                    }
                ]
            },
            {
        ...
    '''
    # Prepare paths
    dataset_root = '/Users/longshen/Code/Datasets/MJN/Preliminary'
    segmented_dir = '/Users/longshen/Code/Datasets/MJN/segmented/5perfs'
    seg_meta_fp = jpath(segmented_dir, 'metadata_seg_debug.json')
    annotation_dir = jpath(dataset_root, 'annotations')
    
    # Get annotation filenames
    annot_fns = [fn for fn in ls(annotation_dir) if not fn.startswith('.')]
    perform_ids = [i.strip().split('.')[0] for i in annot_fns]

    res = {}

    # Iterative over all performances' annotation file
    pbar = tqdm(annot_fns)
    for annot_fn in pbar: # for all performance
        pbar.set_description(annot_fn)
        perform_id = annot_fn.split('.')[0]

        annot_fp = jpath(annotation_dir, annot_fn)
        annot = read_csv_to_json(annot_fp)

        # Normalize the instrument name
        for i, entry in enumerate(annot):
            inst = entry['Description']
            if inst is None:
                continue

            inst = inst.lower()
            inst_stem_name = from_inst_abbr_get_stem_name(inst)
            entry['Description'] = inst_stem_name

        relevant_segments = get_relevant_segments(annot)
        
        all_segmented_data = segment_annotations_based_on_relevance(annot, relevant_segments)

        res[perform_id] = all_segmented_data


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