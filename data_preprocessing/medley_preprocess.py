import os
import sys
import shutil
from pathlib import Path
from tqdm import tqdm

sys.path.append('..')

from utils import ls, jpath, read_json, create_if_not_exist
import torchaudio

def main():
    resample_and_to_mp3_v2()
    # resample_and_to_mp3_v1()

def procedures():
    prepare_meta_for_v2()
    resample_and_to_mp3_v1()
    
    

def resample():
    '''
    Convert the sampling rate to 48000, save with mp3 format
    '''
    pass

def prepare_meta_for_v2():
    '''
    Find the corresponding metadata file from metadata, copy to the dir of each song
    '''
    meta_dir = '/Users/longshen/Code/Datasets/medleydb-meta/medleydb/data/Metadata'
    v2_dir = '/Users/longshen/Code/Datasets/V2_ori'
    songs = [fn for fn in ls(v2_dir) if not fn.startswith('.DS')]
    songs.sort()
    pbar = tqdm(songs)
    for song in pbar:
        pbar.set_description(song)

        song_dir = jpath(v2_dir, song)
        meta_fp = jpath(meta_dir, '{}_METADATA.yaml'.format(song))
        if not os.path.exists(meta_fp):
            raise Exception('Metadata not found for song {}'.format(song))
        shutil.copy(meta_fp, song_dir)


def resample_and_to_mp3_v1():
    '''
    Copy medleydb v1 to a local folder, resample to 48000, save as mp3
    '''
    medleydb_v1_dir = '/Volumes/mixerai/dataset/MedleyDB/Audio'
    mp3_dir = '/Users/longshen/Code/Datasets/MedleyDB_v1_mp3/Audio'
    create_if_not_exist(mp3_dir)

    songs = [fn for fn in ls(medleydb_v1_dir) if not fn.startswith('.')]
    # For each song, copy metadata, mix audio, stem audio
    pbar = tqdm(songs)
    for song in pbar:
        pbar.set_description(song)

        song_dir = jpath(medleydb_v1_dir, song)
        song_new_dir = jpath(mp3_dir, song)
        if os.path.exists(song_new_dir):
            continue
        create_if_not_exist(song_new_dir)

        mix_audio_fp = jpath(song_dir, '{}_MIX.wav'.format(song))
        meta_fp = jpath(song_dir, '{}_METADATA.yaml'.format(song))
        
        # Resample mix audio with torchaudio
        wav, sr_ori = torchaudio.load(mix_audio_fp)
        sr_tgt = 48000
        resampler = torchaudio.transforms.Resample(sr_ori, sr_tgt)
        wav_resampled = resampler(wav)
        torchaudio.save(jpath(song_new_dir, '{}_MIX.mp3'.format(song)), wav_resampled, sr_tgt)

        # Copy metadata
        shutil.copy(meta_fp, song_new_dir)

        # Copy stem audio
        stems_dir = jpath(song_dir, '{}_STEMS'.format(song))
        stems_new_dir = jpath(song_new_dir, '{}_STEMS'.format(song))
        create_if_not_exist(stems_new_dir)
        stems = [fn for fn in ls(stems_dir) if not fn.startswith('.')]
        for stem in tqdm(stems):
            stem_fp = jpath(stems_dir, stem)
            wav, sr_ori = torchaudio.load(stem_fp)
            wav_resampled = resampler(wav)
            torchaudio.save(jpath(stems_new_dir, stem.replace('.wav', '.mp3')), wav_resampled, sr_tgt)

        
def resample_and_to_mp3_v2():
    '''
    Copy medleydb v1 to a local folder, resample to 48000, save as mp3
    '''
    medleydb_v1_dir = '/Users/longshen/Code/Datasets/MedleyDB/original/V2'
    mp3_dir = '/Users/longshen/Code/Datasets/MedleyDB/v2_mp3'
    create_if_not_exist(mp3_dir)

    songs = [fn for fn in ls(medleydb_v1_dir) if not fn.startswith('.')]
    # For each song, copy metadata, mix audio, stem audio
    pbar = tqdm(songs)
    for song in pbar:
        pbar.set_description(song)

        song_dir = jpath(medleydb_v1_dir, song)
        song_new_dir = jpath(mp3_dir, song)
        if os.path.exists(song_new_dir):
            continue
        create_if_not_exist(song_new_dir)

        mix_audio_fp = jpath(song_dir, '{}_MIX.wav'.format(song))
        meta_fp = jpath(song_dir, '{}_METADATA.yaml'.format(song))
        
        # Resample mix audio with torchaudio
        wav, sr_ori = torchaudio.load(mix_audio_fp)
        sr_tgt = 48000
        resampler = torchaudio.transforms.Resample(sr_ori, sr_tgt)
        wav_resampled = resampler(wav)
        torchaudio.save(jpath(song_new_dir, '{}_MIX.mp3'.format(song)), wav_resampled, sr_tgt)

        # Copy metadata
        shutil.copy(meta_fp, song_new_dir)

        # Copy stem audio
        stems_dir = jpath(song_dir, '{}_STEMS'.format(song))
        stems_new_dir = jpath(song_new_dir, '{}_STEMS'.format(song))
        create_if_not_exist(stems_new_dir)
        stems = [fn for fn in ls(stems_dir) if not fn.startswith('.')]
        for stem in tqdm(stems):
            stem_fp = jpath(stems_dir, stem)
            wav, sr_ori = torchaudio.load(stem_fp)
            wav_resampled = resampler(wav)
            torchaudio.save(jpath(stems_new_dir, stem.replace('.wav', '.mp3')), wav_resampled, sr_tgt)


if __name__ == '__main__':
    main()