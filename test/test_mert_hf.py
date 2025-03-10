'''
Test the MERT model by dummy inputs.

Author: Longshen Ou, 2024/07/25
'''

import os
import sys

sys.path.append('.')
sys.path.append('..')

from transformers import Wav2Vec2FeatureExtractor
from transformers import AutoModel
import torch
from torch import nn
import torchaudio.transforms as T
from datasets import load_dataset

from mert.modeling_MERT import MERTModel


def main():
    # from transformers import Wav2Vec2Processor
    mert_playground()


def mert_playground():
    '''
    Workflow:

    '''
    # Loading model, together with some utils from wav2vec to preprocess the audio (mainly do padding and normalization)
    # There might be some warnings about the weights not being used, it's fake. Just ignore.
    model = MERTModel.from_pretrained("m-a-p/MERT-v1-95M")
    audio_processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)

    # Load demo audio and set processor
    dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation", trust_remote_code=True)
    dataset = dataset.sort("id")
    ori_sr = dataset.features["audio"].sampling_rate
    tgt_sr = audio_processor.sampling_rate # 24000 Hz
    resampler = T.Resample(ori_sr, tgt_sr)

    # Resample the audio to the target sampling rate
    if tgt_sr != ori_sr:
        print(f'Resample audio from {ori_sr} to {tgt_sr}')
        t = torch.from_numpy(dataset[0]["audio"]["array"]).float() # [T]

        # Only use the first 5 seconds of the audio
        t = t[:ori_sr*5]

        input_audio = resampler(t) # 140520 samples, 5.855 seconds
    else:
        input_audio = dataset[0]["audio"]["array"]

    # Right side zero pad
    input_audio = nn.functional.pad(input_audio, (0, 320)) # [1, 120000] # pad a frame
    
    # Preprocess the audio
    inputs = audio_processor(input_audio, sampling_rate=tgt_sr, return_tensors="pt") 

    # Forward to the model
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True) # [T=438, D=768], 74.808Hz

    # Take a look at the output shape, there are 13 layers of representation
    # each layer performs differently in different downstream tasks, you should choose empirically
    all_layer_hidden_states = torch.stack(outputs.hidden_states).squeeze()
    print(all_layer_hidden_states.shape) # [13 layer, Time steps, 768 feature_dim]

    # For utterance level classification tasks, you can simply reduce the representation in time
    time_reduced_hidden_states = all_layer_hidden_states.mean(-2)
    print(time_reduced_hidden_states.shape) # [13, 768]

    # You can even use a learnable weighted average representation
    aggregator = nn.Conv1d(in_channels=13, out_channels=1, kernel_size=1)
    weighted_avg_hidden_states = aggregator(time_reduced_hidden_states.unsqueeze(0)).squeeze()
    print(weighted_avg_hidden_states.shape) # [768]


if __name__ == '__main__':
    main()