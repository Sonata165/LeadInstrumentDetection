'''
This file is a instrument "tokenizer"
Which define the mapping between audio filenames, stem names, instrument names, and instrument abbreviations

This version is only compatible with the MJN dataset.

Author: Longshen Ou, 2024/07/25
'''

# Mapping from annotation inst (abbr) to stem name
inst_abbr_to_stem_name = {
    'v': 'vocal',
    'vo': 'vocal',
    'bv': 'back-vocal',
    'sa': 'sax',
    'sa2': 'sax_2',
    'sax': 'sax',
    'sax2': 'sax_2',
    'tr': 'trumpet',
    'br': 'brass',

    'ka': 'kantele',
    'k': 'key',
    'key': 'key',
    'key2': 'key_2',
    'or': 'organ',
    'sy': 'synth',
    'eg': 'e-guitar',
    'eg2': 'e-guitar_2',
    'ag': 'a-guitar',
    'ag2': 'a-guitar_2',
    'ag3': 'a-guitar_3',
    'ag4': 'a-guitar_4',

    'eb': 'e-bass',
    'dr': 'drum',
    'na': 'na',
    'bt': 'pc',
}

# Mapping from stem audio name to stem name
stem_audio_fn_to_stem_name = {
    'background vocal': 'back-vocal',
    'electric guitar': 'e-guitar',
    'bass': 'e-bass',
    'drum': 'drum',
    'mix': 'mix',
    'vocal': 'vocal',
    'brass': 'brass',
    'back-vocal': 'back-vocal',
    'e-guitar': 'e-guitar',
    'key': 'key',
    'e-bass': 'e-bass',
    'synth': 'synth',
    'trumpet': 'trumpet',
    'sax': 'sax',
    'sax2': 'sax_2',
    'keyboard': 'key',
    'pc': 'pc',
    'key_2': 'key_2',
    'e-guitar_2': 'e-guitar_2',
    'organ': 'organ',
    'guitar': 'e-guitar',
    'kantele': 'kantele',
    'cajon': 'cajon',
    'tenor sax': 'sax_2',
    'alto sax': 'sax',
    'acoustic guitar': 'a-guitar',
    'a-guitar': 'a-guitar',
    'acoustic guitar 2': 'a-guitar_2',
    'acoustic guitar 3': 'a-guitar_3',
    'acoustic guitar 4': 'a-guitar_4',
}

# Mapping from stem name to instrument type
stem_inst_map = {
    'vocal': 'vocal',
    'back-vocal': 'vocal',
    'sax': 'sax',
    'sax_2': 'sax',
    'trumpet': 'trumpet',
    'brass': 'brass',

    'kantele': 'kantele',
    'key': 'keyboards',
    'key_2': 'keyboards',
    'organ': 'organ',
    'synth': 'synthesizer',
    'e-guitar': 'electric_guitar',
    'e-guitar_2': 'electric_guitar',
    'a-guitar': 'acoustic_guitar',
    'a-guitar_2': 'acoustic_guitar',
    'a-guitar_3': 'acoustic_guitar',
    'a-guitar_4': 'acoustic_guitar',

    'e-bass': 'electric_bass',
    'drum': 'drum',
    'pc': 'backing_track',
    'cajon': 'cajon',

    'mix': 'mix',
    'na': 'na',
}

medley_inst_name_to_stem_name = {
}

def medley_inst_name_to_stem_name_fn(inst_name):
    '''
    Convert medley inst name to stem name
    '''
    if inst_name in medley_inst_name_to_stem_name:
        return medley_inst_name_to_stem_name[inst_name]
    else:
        raise ValueError('Instrument name not found in medley inst name to stem name mapping')


def main():
    export_inst_name_vocab()

def from_audio_name_get_stem_name(audio_name):
    '''
    Get the stem name from the audio name
    '''
    stem_name = stem_audio_fn_to_stem_name[audio_name]
    return stem_name

def from_audio_name_get_inst_name(audio_name):
    '''
    Get the instrument name from the audio name
    '''
    stem_name = stem_audio_fn_to_stem_name[audio_name]
    inst_name = stem_inst_map[stem_name]
    return inst_name

def from_inst_abbr_get_stem_name(inst_abbr):
    '''
    Get the stem name from the abbreviation
    '''
    stem_name = inst_abbr_to_stem_name[inst_abbr]
    return stem_name

def from_inst_abbr_get_inst_name(inst_abbr):
    '''
    Get the instrument name from the abbreviation
    '''
    stem_name = inst_abbr_to_stem_name[inst_abbr]
    inst_name = stem_inst_map[stem_name]
    return inst_name

def from_stem_name_get_inst_name(stem_name):
    '''
    Get the instrument name from the stem name
    '''
    inst_name = stem_inst_map[stem_name]
    return inst_name


def export_inst_name_vocab():
    '''
    Export the instrument name vocabulary
    '''
    inst_tk = InstTokenizer()
    from utils import save_json
    save_json(inst_tk.id2inst, 'inst_id_to_inst_name.json')


class InstTokenizer:
    '''
    Convert between instrument name (stem name) and id
    '''
    def __init__(self):
        # Compute supported instrument list
        supported_insts = list(set([v for k, v in stem_inst_map.items()]))
        supported_insts.sort()
        self.supported_insts = supported_insts

        # id 0 is reserved for padding
        self.inst2id = {inst: i+1 for i, inst in enumerate(self.supported_insts)}
        self.id2inst = {i+1: inst for i, inst in enumerate(self.supported_insts)}
    
    def convert_inst_to_id(self, inst):
        return self.inst2id[inst]
    
    def convert_id_to_inst(self, idx):
        return self.id2inst[idx]

    def vocab_size(self):
        return len(self.supported_insts)
    
if __name__ == '__main__':
    main()