'''
Recover the song-level annotation from the segment-level ones.
For the MedleyDB dataset
'''
import os
import sys
dirof = os.path.dirname
sys.path.append(dirof(dirof(os.path.abspath(__file__))))

from utils import *
from tqdm import tqdm

def main():
    recover_annotation()


def recover_annotation():
    seg_annot_fp = '/home/longshen/work/LeadInstDetect/LeadInstrumentDetection/datasets/MedleyDB/v1_segmented/metadata_seg.json'
    song_annot_fp = '/home/longshen/work/LeadInstDetect/LeadInstrumentDetection/datasets/MedleyDB/metadata_song.json'
    seg_annot = read_json(seg_annot_fp)

    song_annot = {}
    for song_name, segs_of_song in tqdm(seg_annot.items()):
        annots = []
        for seg in segs_of_song:
            annots_of_seg = seg['annotations']
            for annot in annots_of_seg:
                if annot not in annots:
                    annots.append(annot)
        song_annot[song_name] = annots

    save_json(song_annot, song_annot_fp)




if __name__ == '__main__':
    main()