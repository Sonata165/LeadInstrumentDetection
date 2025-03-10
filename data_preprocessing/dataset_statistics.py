'''
Get some statistics from datasets

Author: Longshen Ou, 2024/07/25
'''

import os
import sys

sys.path.append('..')

from utils import *
import matplotlib.pyplot as plt
import inst_utils_cross_dataset as inst_utils_cd
from inst_utils import from_stem_name_get_inst_name


def main():
    get_duration_medley()


def procedures():
    get_duration()
    get_n_channel_distribution()
    get_lead_inst_dist_by_inst()    # Get lead instrument distribution for MedleyDB dataset

    # For MJN dataset
    get_inst_dist_per_performance() # Get instrument per performance for MJN dataset
    get_lead_inst_dist_by_channel() # Get lead instrument distribution for MJN dataset, by channel
    get_lead_inst_dist_by_inst()    # Get lead instrument distribution for MJN dataset, by instrument

    # For MedleyDB dataset

    # For both
    get_lead_inst_switch_freq()     # Get lead instrument switching frequency
    get_inst_dist()                 # Get all instrument distribution (both lead and non-lead), draw piechart
    visualize_bleed_effect()        # Visualize the bleed effect


def visualize_bleed_effect():
    save_dir = '/home/longshen/work/SoloDetection/misc/bleeding_effect'

    # audio_dir = '/home/longshen/data/datasets/MJN/segmented/5perfs/data/2-1/1'
    # save_fp = jpath(save_dir, 'mjn_example.png')
    # plot_multichannel_audio(audio_dir, save_fp)
    
    audio_dir = '/home/longshen/data/datasets/MedleyDB/v1_segmented/data/AClassicEducation_NightOwl/0'
    save_fp = jpath(save_dir, 'medleydb_example.png')
    plot_multichannel_audio(audio_dir, save_fp)


def plot_multichannel_audio(audio_dir, save_fp):
    '''
    Plot a multitrack heatmap of the audio group
    '''
    # Read audio files
    audio_fns = ls(audio_dir)
    if 'mix.mp3' in audio_fns: # MJN dataset
        mix_fn = 'mix.mp3'
    else:   # MedleyDB dataset
        mix_fn = '0#mix.mp3'
    
    # Put the mix track to the first row
    audio_fns.remove(mix_fn)
    audio_fns = [mix_fn] + audio_fns

    # Read audios, convert to mono, stack together
    audio_fps = [jpath(audio_dir, audio_fn) for audio_fn in audio_fns]
    audios = [torchaudio.load(audio_fp)[0] for audio_fp in audio_fps]
    audios = [convert_waveform_to_mono(audio) for audio in audios]

    # Resample to target sampling rate
    src_sr = 48000
    tgt_sr = 24000
    resampler = torchaudio.transforms.Resample(orig_freq=src_sr, new_freq=tgt_sr)
    audios = [resampler(audio) for audio in audios]
    audio_stack = torch.cat(audios, dim=0).numpy() # [n_channels, n_samples]

    # Get instrument names of each track
    channel_names = [audio_fn.split('.')[0] for audio_fn in audio_fns] # normalized stem names
    inst_names = [inst_utils_cd.from_stem_name_get_inst_name(channel_name) for channel_name in channel_names]

    # Calculate frame-level energy
    audio_sqrt = frame_level_sqrt(audio_stack)[:,:-1]

    plt.figure(figsize=(6,4))
    sns.heatmap(audio_sqrt, cbar=True, cmap='viridis', cbar_kws={'location': 'top'},
                yticklabels=inst_names)
    # plt.title('Bleeding Effect of Multitrack Audio')
    plt.tight_layout()
    plt.savefig(save_fp)


def draw_pie_chart(inst_dist, save_fp):
    # Create an "other" category for instruments with low percentage, sum to 10% together
    other_thresh = 0.09
    other_insts = {}
    sum_perc = 0
    insts = list(inst_dist.keys())
    insts.reverse()
    for inst in insts:
        perc = inst_dist[inst]
        sum_perc += perc
        if sum_perc < other_thresh:
            other_insts[inst] = perc
    for inst in other_insts:
        inst_dist.pop(inst)
    other_perc = sum(other_insts.values())
    inst_dist['other'] = other_perc
    
    import matplotlib.pyplot as plt

    labels = list(inst_dist.keys())
    sizes = list(inst_dist.values())
    explode = [0.1] * len(labels)  # explode
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=False, startangle=90)
    
    # Save figure
    plt.savefig(save_fp)


def get_inst_dist():
    '''
    Obtain the (all) instrument distribution for the dataset
    '''
    def get_inst_dist_from_meta(meta_fp):
        meta = read_json(meta_fp)
        data_root = os.path.dirname(meta_fp)

        res = {}
        for seg_id, entry in meta.items():
            audio_dir = entry['seg_audio_dir']
            audio_path = jpath(data_root, audio_dir)
            audio_names = ls(audio_path)
            for audio_name in audio_names:
                stem_name = audio_name.split('.')[0]
                inst_name = from_stem_name_get_inst_name(stem_name)
                update_dict_cnt(res, inst_name)

        # Remove 'mix' from res
        if 'mix' in res:
            res.pop('mix')

        # Change value to percentage, round to 5th decimal
        total = sum(res.values())
        for k in res:
            res[k] = res[k] / total
            res[k] = round(res[k], 5)
        
        # Sort by value, high to low
        res = dict(sorted(res.items(), key=lambda x: x[1], reverse=True))
            
        return res
    
    meta_fp = '/home/longshen/data/datasets/MedleyDB/v1_segmented/metadata.json'
    res = get_inst_dist_from_meta(meta_fp)
    save_json(res, '/home/longshen/work/SoloDetection/misc/medleyDB/inst_distribution_medleydb.json')

    # Draw a pie chart from res, save to save_dir
    save_dir = '/home/longshen/work/SoloDetection/misc/medleyDB/'
    save_fp = jpath(save_dir, 'inst_distribution_medleydb.png')
    draw_pie_chart(res, save_fp)

    meta_fp = '/home/longshen/data/datasets/MJN/segmented/5perfs/metadata.json'
    res = get_inst_dist_from_meta(meta_fp)
    save_json(res, '/home/longshen/work/SoloDetection/misc/MJN/inst_distribution_mjn.json')

    # Draw a pie chart from res, save to save_dir
    save_dir = '/home/longshen/work/SoloDetection/misc/MJN/'
    save_fp = jpath(save_dir, 'inst_distribution_mjn.png')
    draw_pie_chart(res, save_fp)


def get_lead_inst_switch_freq():
    '''
    Get the lead instrument switching frequency
    '''
    def get_lead_inst_switch_freq_from_meta(meta_fp):
        meta = read_json(meta_fp)
        res = []
        res_dic = {}
        for seg_id, entry in meta.items():
            n_switch = len(entry['annotations']) - 1
            res.append(n_switch)
            update_dict_cnt(res_dic, n_switch)
        print('Average lead instrument switching frequency:', sum(res) / len(res))

        # Sort by key
        res_dic = dict(sorted(res_dic.items(), key=lambda x: x[0]))
        # change value to percentage, round to 5th decimal
        total = sum(res_dic.values())
        for k in res_dic:
            res_dic[k] = res_dic[k] / total
            res_dic[k] = round(res_dic[k], 5)

        return res_dic
    meta_fp = '/home/longshen/data/datasets/MedleyDB/v1_segmented/metadata.json'
    res_dic = get_lead_inst_switch_freq_from_meta(meta_fp)
    save_json(res_dic, '/home/longshen/work/SoloDetection/misc/medleyDB/lead_inst_switch_freq_medleydb.json')

    meta_fp = '/home/longshen/data/datasets/MJN/segmented/5perfs/metadata.json'
    res_dic = get_lead_inst_switch_freq_from_meta(meta_fp)
    save_json(res_dic, '/home/longshen/work/SoloDetection/misc/MJN/lead_inst_switch_freq_mjn.json')


def get_lead_inst_dist_by_inst():
    dist_per_perf = read_json('/home/longshen/work/SoloDetection/misc/lead_inst_distribution_per_performance.json')

    res = {}
    for perf_id, inst_dist in dist_per_perf.items():
        for inst, cnt in inst_dist.items():
            inst_type = from_stem_name_get_inst_name(inst)
            if inst_type not in res:
                res[inst_type] = 0
            res[inst_type] += cnt

    # Sort the result by value
    res = dict(sorted(res.items(), key=lambda x: x[1], reverse=True))

    save_dir = jpath(os.path.dirname(__file__), '..', 'misc')
    save_fp = jpath(save_dir, 'lead_inst_distribution_by_inst.json')
    save_json(res, save_fp)

    # Draw a bar chart with the result
    import matplotlib.pyplot as plt

    x = list(res.keys())
    y = list(res.values())

    plt.bar(x, y)
    plt.xlabel('Lead Instrument (type)')
    plt.ylabel('Count (n_appearance in 5-s segments)')
    plt.title('Lead Instrument Distribution')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(jpath(save_dir, 'lead_inst_distribution_by_inst.png'))


def get_lead_inst_dist_by_channel():
    '''
    Get the lead instrument distribution of the whole dataset
    '''
    dist_per_perf = read_json('/home/longshen/work/SoloDetection/misc/lead_inst_distribution_per_performance.json')

    res = {}
    for perf_id, inst_dist in dist_per_perf.items():
        for inst, cnt in inst_dist.items():
            if inst not in res:
                res[inst] = 0
            res[inst] += cnt

    # Sort the result by value
    res = dict(sorted(res.items(), key=lambda x: x[1], reverse=True))

    save_dir = jpath(os.path.dirname(__file__), '..', 'misc')
    save_fp = jpath(save_dir, 'lead_inst_distribution_by_channel.json')
    save_json(res, save_fp)

    # Draw a bar chart with the result
    import matplotlib.pyplot as plt

    x = list(res.keys())
    y = list(res.values())

    plt.bar(x, y)
    plt.xlabel('Lead Instrument (channel)')
    plt.ylabel('Count (n_appearance in 5-s segments)')
    plt.title('Lead Instrument Distribution')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(jpath(save_dir, 'lead_inst_distribution_by_channel.png'))


def get_inst_dist_per_performance():
    # metadata_fp = '/Users/longshen/Code/Datasets/MJN/segmented/5perfs/metadata.json'
    metadata_fp = '/home/longshen/data/datasets/MJN/segmented/5perfs/metadata.json'
    meta = read_json(metadata_fp)

    res = {}

    for perf_seg_id in tqdm(meta):
        t = perf_seg_id.split('-')
        perf_id = '-'.join(t[:-1])
        seg_id = t[-1]

        if perf_id not in res:
            res[perf_id] = {}
        lead_inst_with_cnt = res[perf_id]

        entry = meta[perf_seg_id]
        lead_insts = [annot['lead'] for annot in entry['annotations']]

        for lead_inst in lead_insts:
            update_dict_cnt(lead_inst_with_cnt, lead_inst)

    # Sort res by key
    res = dict(sorted(res.items(), key=lambda x: int(x[0].split('-')[0]) + 0.1 * int(x[0].split('-')[1])))
    for perf_id, lead_count in res.items():
        # Sort the lead count dict by value, from large to small
        lead_count = dict(sorted(lead_count.items(), key=lambda x: x[1], reverse=True))
        res[perf_id] = lead_count

    save_json(res, 'lead_inst_distribution.json')


def get_lead_inst_dist_by_inst():
    meta = read_json('/Users/longshen/Code/Datasets/MedleyDB/v1_segmented/metadata.json')
    lead_insts = {}
    for seg_id, entry in meta.items():
        for annot in entry['annotations']:
            lead_inst_stem_name = annot['lead']
            lead_inst = inst_utils_cd.from_stem_name_get_inst_name(lead_inst_stem_name)
            update_dict_cnt(lead_insts, lead_inst)
    print(lead_insts)

    # Sort the result by value
    res = dict(sorted(lead_insts.items(), key=lambda x: x[1], reverse=True))

    save_dir = '/Users/longshen/Code/SoloDetection/misc/medleyDB'
    save_fp = jpath(save_dir, 'lead_inst_distribution_by_inst.json')
    save_json(res, save_fp)

    # Draw a bar chart with the result
    import matplotlib.pyplot as plt

    x = list(res.keys())
    y = list(res.values())

    plt.bar(x, y)
    plt.xlabel('Lead Instrument (type)')
    plt.ylabel('Count (n_appearance in 5-s segments)')
    plt.title('Lead Instrument Distribution')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(jpath(save_dir, 'lead_inst_distribution_by_inst.png'))


def get_duration_medley():
    metadata_fp = '/home/longshen/data/datasets/MedleyDB/v1_segmented/metadata.json'
    
    meta = read_json(metadata_fp)
    tot_dur = len(meta) * 5 / 2 # 5 s per segment, 2.5 s overlap
    print('---------------- Total -----------------')
    print(f'duration: {tot_dur/3600:.2f} h')
    print('Number of performances: 25')
    print(f'number of segments: {len(meta)}')
    print()

    # Get the n_performance, n_segments, durations, of each split
    splits = ['train', 'valid', 'test']
    for split in splits:
        split_meta = {k:v for k,v in meta.items() if v['split'] == split}
        n_perfs = len(split_meta)
        n_segs = len(split_meta)
        perf_ids = set(['-'.join(k.split('-')[:-1]) for k in split_meta.keys()])
        
        n_perfs = len(perf_ids)
        print('----------------- {} ----------------'.format(split))
        print(perf_ids)
        print(f'{n_perfs} performances, {n_segs} segments')
        print(f'{split} set duration: {n_segs * 5 / 2 / 3600:.2f} h')
        print()


def get_duration():
    metadata_fp = '/home/longshen/data/datasets/MJN/segmented/5perfs/metadata.json'
    
    meta = read_json(metadata_fp)
    tot_dur = len(meta) * 5 / 2 # 5 s per segment, 2.5 s overlap
    print('---------------- Total -----------------')
    print(f'duration: {tot_dur/3600:.2f} h')
    print('Number of performances: 25')
    print(f'number of segments: {len(meta)}')
    print()

    # Get the n_performance, n_segments, durations, of each split
    splits = ['train', 'valid', 'test']
    for split in splits:
        split_meta = {k:v for k,v in meta.items() if v['split'] == split}
        n_perfs = len(split_meta)
        n_segs = len(split_meta)
        perf_ids = set(['-'.join(k.split('-')[:-1]) for k in split_meta.keys()])
        
        n_perfs = len(perf_ids)
        print('----------------- {} ----------------'.format(split))
        print(perf_ids)
        print(f'{n_perfs} performances, {n_segs} segments')
        print(f'{split} set duration: {n_segs * 5 / 2 / 3600:.2f} h')
        print()


def get_n_channel_distribution():
    data_dir = '/home/longshen/data/datasets/MJN/segmented/5perfs/data'

    cnt_dic = {}
    pfm_ids = ls(data_dir)
    for pfm_id in pfm_ids:
        pfm_dir = os.path.join(data_dir, pfm_id)
        seg_ids = ls(pfm_dir)
        for seg_id in seg_ids:
            seg_dir = os.path.join(pfm_dir, seg_id)
            n_channels = len(ls(seg_dir))
            update_dict_cnt(cnt_dic, n_channels)

    # Sort by key
    cnt_dic = dict(sorted(cnt_dic.items(), key=lambda item: item[0]))
    save_dir = '/home/longshen/work/SoloDetection/misc'

    # Draw bar chart
    x = list(cnt_dic.keys())
    y = list(cnt_dic.values())
    plt.bar(x, y)
    plt.xlabel('Number of channels')
    plt.ylabel('Number of segments')
    plt.title('Number of channels distribution')
    plt.show()
    plt.savefig(os.path.join(save_dir, 'n_channels_distribution.png'))

    # Save dict
    save_json(cnt_dic, os.path.join(save_dir, 'n_channels_distribution.json'))



if __name__ == '__main__':
    main()