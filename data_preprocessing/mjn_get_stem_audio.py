import os
import sys

sys.path.append('.')
sys.path.append('..')

import json
import pandas as pd
import torch
import torchaudio
from utils import jpath, create_if_not_exist, ls, convert_waveform_to_mono
from tqdm import tqdm
import re

def main():
    generate_stem_audio('14-4')
    # procedures()

def procedures():
    generate_stem_audio()

def generate_stem_audio(pfm_band_id):
    '''
    Generate stem audio for a given performance.
    '''
    print('Generating stem audio for performance {}'.format(pfm_band_id))
    pfm_id, band_id = pfm_band_id.split('-')

    # Prepare configs
    data_root = '/Users/longshen/Code/Datasets/MJN/Preliminary'
    target_sr = 48000
    save_type = 'mp3'

    meta_dir = jpath(data_root, 'metadata')
    meta_raw_audio_dir_fn = 'raw_audio_dirs.xlsx'
    inst_map_dir = jpath(data_root, 'inst_map') # root dir of channel-to-inst mapping

    # Read metadata, obtain raw audio's location
    meta_raw_audio_dir_fp = jpath(meta_dir, meta_raw_audio_dir_fn)
    meta_raw_audio_dir = read_excel_to_json(meta_raw_audio_dir_fp) # class: pd.DataFrame

    if pfm_band_id not in meta_raw_audio_dir:
        raise Exception('Performance ID {} not exists in metadata'.format(pfm_band_id))
    raw_audio_dir = meta_raw_audio_dir[pfm_band_id]['Raw Audio Dir']

    # Obtain mapping between channel and instrument names
    inst_map_fp = jpath(inst_map_dir, '{}.xlsx'.format(pfm_band_id))
    if os.path.exists(inst_map_fp) is True:
        inst_map = read_excel_to_json(inst_map_fp)
        
    else:
        mjn_id = pfm_band_id.split('-')[0]
        print('{} does not exist, use MJN{}.xlsx instead'.format(inst_map_fp, mjn_id))
        inst_map_fp = jpath(inst_map_dir, 'MJN{}.xlsx'.format(mjn_id))
        inst_map = read_excel_to_json(inst_map_fp)

    tracks_of_inst = {}
    
    # Note: actual audio tracks may be less than that in metadata. Iterate over actual audio tracks
    track_fns = [fn for fn in ls(raw_audio_dir) if not fn.startswith('.')]
    
    
    ''' Ensure all audios in raw audio dir are defined in inst mapping '''
    print('Legality check ...')
    track_names = {}
    stopwords = [ # For audios containing below words, ignore them.
        'Spare', 'Audio', 'Interview', '29fps', 'Rec', '1280x720', 'beatlooser.wav',
        'Ambient', 'Audience'
    ]
    if pfm_id == '15':    
        stopwords.append('FoH')

    for i, track_fn in enumerate(track_fns):
        if pfm_id == '15':
            t = '.'.join(track_fn.split('.')[:-1]) # Remove file extension
            t = t.split('_')[2:4]   # Get the track name
            track_name = '_'.join(t) + '.wav'   # Add back the file extension
            
        elif pfm_id == '14':
            matches = re.findall(r'\d{1}\D*\d{4}', track_fn)
            if matches:
                track_name = track_fn.replace(matches[0], '')[1:] # remove the '_' in the beginning
            else:
                raise Exception('Cannot find track name in {}'.format(track_fn))

            # t = '.'.join(track_fn.split('.')[:-1])
            # t = t.split('_')[3:]
            # track_name = '_'.join(t) + '.wav'
        else:
            track_name = track_fn

        if track_name not in inst_map:  
            # Skip files whose track name contains stopwords
            if any([stopword in track_fn for stopword in stopwords]):
                continue

            # Ensure the track is not empty
            raise Exception('Track {} with name {} not defined in inst mapping'.format(track_fn, track_name))
        
        # Skip audios that has empty stem assignment
        if inst_map[track_name]['Instrument'] is None:
            continue

        # Save the track name of each audio file
        track_names[track_fn] = track_name
        
    # Ensure there is mix-l and mix-r in actual audio tracks
    # Get the filename of track that is mix-l and mix-r
    mix_l_fn = [fn for fn in track_names if inst_map[track_names[fn]]['Instrument'] == 'mix-l']
    mix_r_fn = [fn for fn in track_names if inst_map[track_names[fn]]['Instrument'] == 'mix-r']
    if len(mix_l_fn) != 1:
        raise Exception('Found {} mix-l track in {}\ntrack_names: {}'.format(len(mix_l_fn), pfm_band_id, track_names.keys()))
    if len(mix_r_fn) != 1:
        raise Exception('Found {} mix-r track in {}'.format(len(mix_r_fn), pfm_band_id))
    
    # Read raw audios, merge them to a single stereo audio, normalize volume
    # Iterate over all used instruments
    print('Reading raw audio ...')
    pbar = tqdm(track_fns)
    for track_fn in pbar:
        pbar.set_description(track_fn)
        if track_fn not in track_names: # Skip non-instrumental tracks
            continue

        track_name = track_names[track_fn]
        track_fp = jpath(raw_audio_dir, track_fn)

        # Skip files whose track name contains stopwords
        if any([sw in track_name for sw in stopwords]):
            continue
        
        # Ensure the track is not empty
        inst = inst_map[track_name]['Instrument']
        # if inst == 'mix-l':
        #     a=2
        if inst == None: # Tracks that doesn't have instrument label, are empty
            continue
        if inst not in tracks_of_inst:
            tracks_of_inst[inst] = []

        # Ensure the track audio exist
        if os.path.exists(track_fp) is False:
            raise Exception('Track {} not exists in raw audio dir {}'.format(track_fp, raw_audio_dir))
        
        try:
            track_audio, sr = torchaudio.load(track_fp, format='mp3')
        except:
            # Skip any damaged audio
            print('Error loading track {}'.format(track_fn))
            continue

        if sr != target_sr:
            # Instantiate a Resample transform
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
            
            # Resample the waveform
            track_audio = resampler(track_audio)
        tracks_of_inst[inst].append(track_audio)
    
    # Delete empty inst from tracks_of_inst (may due to empty audio tracks)
    empty_inst = []
    for inst in tracks_of_inst:
        if len(tracks_of_inst[inst]) == 0:
            empty_inst.append(inst)
    for inst in empty_inst:
        del tracks_of_inst[inst]

    # Average raw data of each instrument to get stem audio
    print('Collating mix audio ...')
    mix_l = tracks_of_inst['mix-l'][0] # mono audio in torch tensor, shape: [n_samples]
    mix_l = convert_waveform_to_mono(mix_l)
    mix_r = tracks_of_inst['mix-r'][0]
    mix_r = convert_waveform_to_mono(mix_r)
    mix = torch.cat([mix_l, mix_r], dim=0)
    # mix = torch.stack([mix_l, mix_r])
    mix_normalized = normalize_waveform(mix)
    out_dir = jpath(data_root, 'stem_audio', pfm_band_id)
    create_if_not_exist(out_dir)
    out_fp = jpath(out_dir, 'mix.mp3')
    torchaudio.save(out_fp, mix_normalized, target_sr)

    # Delete mix-l and mix-r from tracks_of_inst
    del tracks_of_inst['mix-l']
    del tracks_of_inst['mix-r']

    # Average raw data of each instrument to get stem audio
    print('Collating ...')
    for inst in tracks_of_inst:
        t = torch.stack(tracks_of_inst[inst])
        # Average all audio of a same instrument
        avg_waveform = torch.mean(t, dim=0)

        # If avg_waveform is stereo, convert to mono
        avg_waveform = convert_waveform_to_mono(avg_waveform)

        normalized_waveform = normalize_waveform(avg_waveform)

        # # Stop normalize the waveform
        # normalized_waveform = avg_waveform

        # Save the stem audio to desired directory: data_root/stem_audio/perf_id/{inst}.wav
        out_dir = jpath(data_root, 'stem_audio', pfm_band_id)
        create_if_not_exist(out_dir)
        out_fp = jpath(out_dir, '{}.{}'.format(inst, save_type))
        torchaudio.save(out_fp, normalized_waveform, target_sr)


def generate_all_stem_audio():
    # Prepare configs
    data_root = '/Users/longshen/Code/Datasets/MJN/Preliminary'
    target_sr = 48000
    save_type = 'mp3'

    meta_dir = jpath(data_root, 'metadata')
    meta_raw_audio_dir_fn = 'raw_audio_dirs.xlsx'
    inst_map_dir = jpath(data_root, 'inst_map') # root dir of channel-to-inst mapping

    # Read metadata, obtain raw audio's location
    meta_raw_audio_dir_fp = jpath(meta_dir, meta_raw_audio_dir_fn)
    meta_raw_audio_dir = read_excel_to_json(meta_raw_audio_dir_fp) # class: pd.DataFrame

    # Iterate over all performance entries
    pbar = tqdm(meta_raw_audio_dir)
    for performance_id in pbar:
        pbar.set_description(performance_id)
        raw_audio_dir = meta_raw_audio_dir[performance_id]['Raw Audio Dir']

        # Obtain mapping between channel and instrument names
        inst_map_fp = jpath(inst_map_dir, '{}.xlsx'.format(performance_id))
        inst_map = read_excel_to_json(inst_map_fp)

        tracks_of_inst = {}
        # Read raw audios, merge them to a single stereo audio, normalize volume
        # Iterate over all used instruments
        print('Reading raw audio ...')
        for track_name in inst_map:
            track_fp = jpath(raw_audio_dir, track_name)
            
            # Ensure the track is not empty
            inst = inst_map[track_name]['Instrument']
            if inst == None: # Tracks that doesn't have instrument label, are empty
                continue
            if inst not in tracks_of_inst:
                tracks_of_inst[inst] = []

            # Ensure the track audio exist
            if os.path.exists(track_fp) is False:
                raise Exception('Track {} not exists in raw audio dir {}'.format(track_fp, raw_audio_dir))
            
            
            track_audio, sr = torchaudio.load(track_fp)
            if sr != target_sr:
                # Instantiate a Resample transform
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
                
                # Resample the waveform
                track_audio = resampler(track_audio)
            tracks_of_inst[inst].append(track_audio)
        
        # Average raw data of each instrument to get stem audio
        print('Collating mix audio ...')
        mix_l = tracks_of_inst['mix-l']
        mix_r = tracks_of_inst['mix-r']
        mix = torch.stack([mix_l, mix_r])
        mix_normalized = normalize_waveform(mix)
        out_dir = jpath(data_root, 'stem_audio', performance_id)
        create_if_not_exist(out_dir)
        out_fp = jpath(out_dir, 'mix.mp3')
        torchaudio.save(out_fp, mix_normalized, target_sr)

        # Delete mix-l and mix-r from tracks_of_inst
        del tracks_of_inst['mix-l']
        del tracks_of_inst['mix-r']

        print('Collating ...')
        for inst in tracks_of_inst:
            t = torch.stack(tracks_of_inst[inst])
            avg_waveform = torch.mean(t, dim=0)

            # If avg_waveform is stereo, convert to mono
            if avg_waveform.shape[0] == 2:
                avg_waveform = torch.mean(avg_waveform, dim=0)

            # Normalize
            normalized_waveform = normalize_waveform(avg_waveform)

            # Save the stem audio to desired directory: data_root/stem_audio/perf_id/{inst}.wav
            out_fp = jpath(out_dir, '{}.{}'.format(inst, save_type))
            torchaudio.save(out_fp, normalized_waveform, target_sr)

    return None

def normalize_waveform(waveform, target_db=-0.1):
    """
    Normalize the waveform to a specific dB level.

    Parameters:
    waveform (torch.Tensor): The input waveform tensor.
    target_db (float): The target dB level for normalization.

    Returns:
    torch.Tensor: The normalized waveform.
    """
    # Calculate the target amplitude
    target_amplitude = 10 ** (target_db / 20)

    # Find the peak amplitude in the waveform
    peak_amplitude = torch.max(torch.abs(waveform))

    # Calculate the scaling factor
    scaling_factor = target_amplitude / peak_amplitude

    # Normalize the waveform
    normalized_waveform = waveform * scaling_factor

    return normalized_waveform



def read_excel_to_json(excel_fp):
    '''
    Read excel file with pandas to DataFrame,
    Then convert to json.
    Format: [{column -> value}, … , {column -> value}]
    '''
    meta_raw_audio_dir = pd.read_excel(excel_fp, index_col=0) # class: pd.DataFrame
    t = meta_raw_audio_dir.transpose().to_json()
    ret = json.loads(t)
    return ret







if __name__ == '__main__':
    main()