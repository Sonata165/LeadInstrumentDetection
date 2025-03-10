'''
Pytorch models for solo detection

Author: Longshen Ou, 2024/07/25
'''

import torch
import torchaudio
import torch.nn as nn
from mert.modeling_MERT import MERTModel


def get_model(model_config):
    model_class = eval(model_config.model_class)
    model = model_class(model_config)
    return model

 
class MertSoloDetectorFromMix(nn.Module):
    '''
    From Mix model
    Instrument classification
    '''
    def __init__(self = None, model_config = None):
        super().__init__()
        self.config = model_config
        self.mert = MERTModel.from_pretrained(model_config.mert_url)
        self.inst_cls_head = nn.Linear(768, model_config.num_inst_type + 1)
 

    def forward(self, x, insts=None):
        bs, n_ch, n_samples = x.size()   # [bs, n_ch, n_samples]
        x = x.view(-1, x.size(-1))       # [bs*n_ch, n_samples]
        x = self.mert(x).last_hidden_state   # [bs*n_ch, n_frame, n_dim]
        _, n_frame, n_dim = x.size()        
        x = x.view(bs, n_ch, n_frame, n_dim)    # [bs, n_ch, n_frame, n_dim]
        
        mix_channel_id = self.config['mix_ch_id']
        mix_idx = (insts == mix_channel_id).nonzero(as_tuple=True)[1]
        mix_x = x[torch.arange(bs), mix_idx, :, :] # [bs, n_frame, n_dim]
        x = mix_x
        x = self.inst_cls_head(x)
        return x


class MertSoloDetectorFromMixSegment(nn.Module):
    '''
    From Mix model
    Segment-level
    Binary classification (guitar / non-guitar)
    '''
    def __init__(self = None, model_config = None):
        super().__init__()
        self.config = model_config
        self.mert = MERTModel.from_pretrained(model_config.mert_url)
        self.inst_cls_head = nn.Linear(768, model_config.num_inst_type) # Is guitar solo, or not
 

    def forward(self, x, insts=None):
        bs, n_ch, n_samples = x.size()   # [bs, n_ch, n_samples]

        x = x.squeeze(1) # [bs, n_samples]
        x = self.mert(x).last_hidden_state   # [bs, n_frame, n_dim]
        _, n_frame, n_dim = x.size()        

        # Average along the time axis
        x = x.mean(dim=1) # [bs, n_dim]

        # Classification        
        x = self.inst_cls_head(x)

        return x
 
 
class MertSoloDetectorByInst(nn.Module):
    '''
    Channel average model without instrument embedding
    Instrument classification
    '''
    def __init__(self = None, model_config = None):
        super().__init__()
        self.config = model_config
        self.mert = MERTModel.from_pretrained(model_config.mert_url)
        self.inst_cls_head = nn.Linear(768, model_config.num_inst_type + 1)
 

    def forward(self, x):
        (bs, max_inst_num, n_samples) = x.size()
        x = x.view(-1, x.size(-1))
        x = self.mert(x).last_hidden_state
        (_, n_frame, n_dim) = x.size() # [bs*n_ch, n_frame, n_dim]
        x = x.view(bs, max_inst_num, n_frame, n_dim)
        x = x.permute(0, 2, 1, 3) # [bs, n_frame, n_ch, n_dim]
        x = x.mean(2, **('dim',))
        x = self.inst_cls_head(x)
        return x
 
 
class MertSoloDetectorInstEmb(nn.Module):
    '''
    The "channel average model" with instrument embedding
    Instrument classification
    '''
    def __init__(self = None, model_config = None):
        super().__init__()
        self.config = model_config
        self.mert = MERTModel.from_pretrained(model_config.mert_url)
        self.inst_cls_head = nn.Linear(768, model_config.num_inst_type + 1)
        self.inst_emb = nn.Embedding(model_config.num_inst_type + 1, 768)
 

    def forward(self, x, insts = (None,)):
        (bs, max_inst_num, n_samples) = x.size()
        x = x.view(-1, x.size(-1))
        x = self.mert(x).last_hidden_state
        (_, n_frame, n_dim) = x.size()
        x = x.view(bs, max_inst_num, n_frame, n_dim) # [bs, n_ch, n_frame, n_dim]
        inst_emb = self.inst_emb(insts) # [bs, n_ch, n_dim]
        x = x + inst_emb.unsqueeze(2)
        x = x.permute(0, 2, 1, 3)   # [bs, n_frame, n_ch, n_dim]
        x = x.mean(dim=2)           # [bs, n_frame, n_dim]
        x = self.inst_cls_head(x)
        return x
 
 
class MertSoloDetectorByChannel(nn.Module):
    '''
    The model with channel-wise attention
    Do the instrument classification
    '''
    def __init__(self = None, model_config = None):
        super().__init__()
        self.config = model_config
        self.mert = MERTModel.from_pretrained(model_config.mert_url)
        self.inst_cls_head = nn.Linear(768, model_config.num_inst_type + 1)
        self.inst_emb = nn.Embedding(model_config.num_inst_type + 1, 768)
        self.channel_attn = nn.MultiheadAttention(embed_dim=768, num_heads=model_config.attn_heads, batch_first=True)
        self.attn_drop = nn.Dropout(model_config.attn_drop) # zero out attn_drop% of the attention weights
 

    def forward(self, x, insts = (None,)):
        (bs, n_ch, n_samples) = x.size() # [bs, n_ch, n_samples]
        x = x.view(-1, x.size(-1))  # [bs*n_ch, n_samples]
        x = self.mert(x).last_hidden_state # [bs*n_ch, n_frame, n_dim]
        (_, n_frame, n_dim) = x.size()
        x = x.view(bs, n_ch, n_frame, n_dim)    # [bs, n_ch, n_frame, n_dim]

        # Add Instrument Embedding
        inst_emb = self.inst_emb(insts)
        x = x + inst_emb.unsqueeze(2) # [bs, n_ch, n_frame, n_dim]

        # Prepare the mixture channel
        mix_channel_id = self.config['mix_ch_id']
        mix_idx = (insts == mix_channel_id).nonzero(as_tuple=True)[1]
        mix_x = x[torch.arange(bs), mix_idx, :, :].unsqueeze(1)
        mix_x = mix_x.view(bs * n_frame, n_dim).unsqueeze(1) # [bs*n_frame, 1, n_dim]

        # Prepare other channels as k and v
        x = x.permute(0, 2, 1, 3) # [bs, n_frame, n_ch, n_dim]
        x = x.reshape(bs * n_frame, n_ch, n_dim)    # [bs*n_frame, n_ch, n_dim]

        # Channel-wise attention
        (x, _) = self.channel_attn(mix_x, x, x) # [bs*n_frame, 1, n_dim]
        x = self.attn_drop(x)

        # Classification
        x = x.squeeze(1)    # [bs*n_frame, n_dim]
        x = x.view(bs, n_frame, n_dim)  # [bs, n_frame, n_dim]
        x = self.inst_cls_head(x) # [bs, n_frame, n_inst_type]

        return x
    

class MertSoloDetectorChattnChcls(nn.Module):
    '''
    The model with channel-wise attention
    For channel classification setting
    '''
    def __init__(self = None, model_config = None):
        super().__init__()
        self.config = model_config
        self.mert = MERTModel.from_pretrained(model_config.mert_url)
        self.ch_cls_head = nn.Linear(768, model_config.max_channel_num)
        self.inst_emb = nn.Embedding(model_config.num_inst_type + 1, 768)
        self.ch_emb = nn.Embedding(model_config.max_channel_num + 1, 768)
        self.channel_attn = nn.MultiheadAttention(embed_dim=768, num_heads=model_config.attn_heads, batch_first=True)
        self.attn_drop = nn.Dropout(model_config.attn_drop) # zero out attn_drop% of the attention weights

        self.ln = nn.LayerNorm(768)
        self.ln2 = nn.LayerNorm(768)
 

    def forward(self, x, insts = (None,)):
        (bs, n_ch, n_samples) = x.size() # [bs, n_ch, n_samples]
        x = x.view(-1, x.size(-1))  # [bs*n_ch, n_samples]
        x = self.mert(x).last_hidden_state # [bs*n_ch, n_frame, n_dim]
        (_, n_frame, n_dim) = x.size()
        x = x.view(bs, n_ch, n_frame, n_dim)    # [bs, n_ch, n_frame, n_dim]

        # Add Instrument Embedding
        if self.config.get('inst_emb', True) is True:
            inst_emb = self.inst_emb(insts) # [bs, n_ch, n_dim]
            x = x + inst_emb.unsqueeze(2)

        # Add Channel Embedding
        if self.config.get('ch_emb', True) is True:
            ch_emb = self.ch_emb(torch.arange(n_ch, device=x.device)) # [n_ch, n_dim]
            x = x + ch_emb.unsqueeze(1)

        # Prepare the mixture channel
        mix_channel_id = self.config['mix_ch_id']
        mix_x = x[:, mix_channel_id, :, :].unsqueeze(1) # [bs, 1, n_frame, n_dim]
        mix_x = mix_x.permute(0, 2, 1, 3).reshape(-1, 1, n_dim) # [bs * n_frame, 1, n_dim]

        # Prepare other channels as k and v
        x = x.permute(0, 2, 1, 3) # [bs, n_frame, n_ch, n_dim]
        x = x.reshape(bs * n_frame, n_ch, n_dim)    # [bs*n_frame, n_ch, n_dim]

        # Channel-wise attention
        (x, _) = self.channel_attn(mix_x, x, x) # [bs*n_frame, 1, n_dim]

        # Layer norm
        x = self.ln(x)

        x = self.attn_drop(x)

        # Classification
        x = x.squeeze(1)    # [bs*n_frame, n_dim]
        x = x.view(bs, n_frame, n_dim)  # [bs, n_frame, n_dim]
        x = self.ch_cls_head(x) # [bs, n_frame, n_inst_type]

        return x
    

class MertSoloDetectorChattnChclsNoMix(nn.Module):
    '''
    The model with channel-wise attention 
    For channel classification setting

    This variant does not use the mixture channel's info.
    Instead, it create a mixture channel itself by averaging all other channels.
    '''
    def __init__(self = None, model_config = None):
        super().__init__()
        self.config = model_config
        self.mert = MERTModel.from_pretrained(model_config.mert_url)
        self.ch_cls_head = nn.Linear(768, model_config.max_channel_num)
        self.inst_emb = nn.Embedding(model_config.num_inst_type + 1, 768)
        self.ch_emb = nn.Embedding(model_config.max_channel_num + 1, 768)
        self.channel_attn = nn.MultiheadAttention(embed_dim=768, num_heads=model_config.attn_heads, batch_first=True)
        self.attn_drop = nn.Dropout(model_config.attn_drop) # zero out attn_drop% of the attention weights

        self.ln = nn.LayerNorm(768)
 

    def forward(self, x, insts = (None,)):
        (bs, n_ch, n_samples) = x.size() # [bs, n_ch, n_samples]
        x = x.view(-1, x.size(-1))  # [bs*n_ch, n_samples]
        x = self.mert(x).last_hidden_state # [bs*n_ch, n_frame, n_dim]
        (_, n_frame, n_dim) = x.size()
        x = x.view(bs, n_ch, n_frame, n_dim)    # [bs, n_ch, n_frame, n_dim]

        # Add Instrument Embedding
        inst_emb = self.inst_emb(insts) # [bs, n_ch, n_dim]
        x = x + inst_emb.unsqueeze(2)

        # Add Channel Embedding
        ch_emb = self.ch_emb(torch.arange(n_ch, device=x.device)) # [n_ch, n_dim]
        x = x + ch_emb.unsqueeze(1)

        # Prepare to create mixture channel
        mixture_ch_id = 0
        other_channels = x[:, 1:, :, :] # [bs, n_ch-1, n_frame, n_dim]
        mix_x = other_channels.mean(dim=1, keepdim=True) # [bs, 1, n_frame, n_dim]
        mix_x = mix_x.permute(0, 2, 1, 3).reshape(-1, 1, n_dim) # [bs * n_frame, 1, n_dim]

        # Prepare other channels as k and v
        x = x.permute(0, 2, 1, 3) # [bs, n_frame, n_ch, n_dim]
        x = x.reshape(bs * n_frame, n_ch, n_dim)    # [bs*n_frame, n_ch, n_dim]

        # Channel-wise attention
        (x, _) = self.channel_attn(mix_x, x, x) # [bs*n_frame, 1, n_dim]

        # Layer norm
        x = self.ln(x)

        x = self.attn_drop(x)

        # Classification
        x = x.squeeze(1)    # [bs*n_frame, n_dim]
        x = x.view(bs, n_frame, n_dim)  # [bs, n_frame, n_dim]
        x = self.ch_cls_head(x) # [bs, n_frame, n_inst_type]

        return x


class MultiChannelCRNN(nn.Module):
    '''
    Baseline model from 

    Adavanne, Sharath, Archontis Politis, and Tuomas Virtanen. 
    "Multichannel sound event detection using 3D convolutional neural networks for learning inter-channel features." 
    In 2018 international joint conference on neural networks (IJCNN), pp. 1-7. IEEE, 2018.

    Original repo: https://github.com/sharathadavanne/sed-crnn?tab=readme-ov-file
    '''
    def __init__(self, model_config=None):
        '''
        input_shape, output_dim, cnn_nb_filt, cnn_pool_size, rnn_nb, fc_nb, dropout_rate
        '''
        super().__init__()

        self.transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=24000,
            n_fft=2048,
            hop_length=320,
        )

        # There are 10 input channels at most (include mixture channel)

        input_shape = [10, 375, 2048] # [n_ch, n_frame, n_dim]
        output_dim = 16 + 1 # Same as my inst_cls model, n_inst + 1 for na
        cnn_nb_filt = 128 # From the repo
        cnn_pool_size = [5, 2, 2]
        rnn_nb = [32, 32]
        rnn_hidden_size = 32
        fc_nb = [32]
        dropout_rate = 0.

        self.conv_blocks = nn.ModuleList()
        self.rnn_blocks = nn.ModuleList()
        self.fc_blocks = nn.ModuleList()

        current_channels = input_shape[0]
        current_length = input_shape[1]

        # Convolutional layers
        for i, pool_size in enumerate(cnn_pool_size):
            self.conv_blocks.append(nn.Sequential(
                nn.Conv2d(current_channels, cnn_nb_filt, kernel_size=(3, 3), padding='same'),
                nn.BatchNorm2d(cnn_nb_filt),
                nn.ReLU(),
                nn.MaxPool2d((1, pool_size)),
                nn.Dropout(dropout_rate)
            ))
            current_channels = cnn_nb_filt  # update the channel number for the next layer

        # RNN layers
        self.rnn = nn.GRU(input_size=768,
                            hidden_size=rnn_hidden_size, 
                            num_layers=2, 
                            batch_first=True, 
                            bidirectional=True, 
                            dropout=dropout_rate
                            )

        # Fully connected layers
        for f in fc_nb:
            self.fc_blocks.append(nn.Sequential(
                nn.Linear(rnn_hidden_size * 2, f),  # *2 because of bidirectional
                nn.Dropout(dropout_rate)
            ))
            r = f  # next input features will be the output of the current layer

        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(r, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x, insts=None):
        '''
        x: [bs, n_ch, n_frame, n_dim]
        '''

        x = x[:, :, :119999] # truncate so that spectrogram is 75Hz
        x = torch.permute(self.transform(x), (0, 1, 3, 2)) # [bs, n_ch, n_frame=375, n_dim=128 (mel bins)]

        # Pad channels to maximum
        n_ch = x.size(1)
        tgt_ch = 10
        if n_ch < tgt_ch:
            x = torch.cat([x, torch.zeros(x.size(0), tgt_ch - n_ch, x.size(2), x.size(3), device=x.device)], dim=1)

        for conv in self.conv_blocks:
            x = conv(x)
        # [bs, n_ch_cnn, n_frame, n_dim_cnn_out]

        x = torch.permute(x, (0,2,1,3))  # [bs, n_frame, n_ch_cnn, n_dim_cnn_out]
        x = x.reshape(x.size(0), x.size(1), -1) # [bs, n_frame, cnn_dim = n_ch_cnn * n_dim_cnn_out = 768]

        x, _ = self.rnn(x)

        x = x.contiguous()
        for fc in self.fc_blocks:
            x = fc(x)

        x = self.output_layer(x)
        return x



class MultiChannelMertCRNN(nn.Module):
    '''
    CRNN with MERT features as input
    With or without MERT finetune
    '''
    def __init__(self, model_config=None):
        '''
        input_shape, output_dim, cnn_nb_filt, cnn_pool_size, rnn_nb, fc_nb, dropout_rate
        '''
        super().__init__()

        self.mert = MERTModel.from_pretrained(model_config.mert_url)

        self.transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=24000,
            n_fft=2048,
            hop_length=320,
        )

        # There are 10 input channels at most (include mixture channel)

        input_shape = [10, 375, 2048] # [n_ch, n_frame, n_dim]
        output_dim = 16 + 1 # Same as my inst_cls model, n_inst + 1 for na
        cnn_nb_filt = 128 # From the repo
        cnn_pool_size = [5, 2, 2]
        rnn_nb = [32, 32]
        rnn_hidden_size = 32
        fc_nb = [32]
        dropout_rate = 0.

        self.conv_blocks = nn.ModuleList()
        self.rnn_blocks = nn.ModuleList()
        self.fc_blocks = nn.ModuleList()

        current_channels = input_shape[0]
        current_length = input_shape[1]

        # Convolutional layers
        for i, pool_size in enumerate(cnn_pool_size):
            self.conv_blocks.append(nn.Sequential(
                nn.Conv2d(current_channels, cnn_nb_filt, kernel_size=(3, 3), padding='same'),
                nn.BatchNorm2d(cnn_nb_filt),
                nn.ReLU(),
                nn.MaxPool2d((1, pool_size)),
                nn.Dropout(dropout_rate)
            ))
            current_channels = cnn_nb_filt  # update the channel number for the next layer

        # RNN layers
        self.rnn = nn.GRU(input_size=4864,
                            hidden_size=rnn_hidden_size, 
                            num_layers=2, 
                            batch_first=True, 
                            bidirectional=True, 
                            dropout=dropout_rate
                            )

        # Fully connected layers
        for f in fc_nb:
            self.fc_blocks.append(nn.Sequential(
                nn.Linear(rnn_hidden_size * 2, f),  # *2 because of bidirectional
                nn.Dropout(dropout_rate)
            ))
            r = f  # next input features will be the output of the current layer

        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(r, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x, insts=None):
        '''
        x: [bs, n_ch, n_frame, n_dim]
        '''

        (bs, max_inst_num, n_samples) = x.size()
        x = x.view(-1, x.size(-1))
        x = self.mert(x).last_hidden_state
        (_, n_frame, n_dim) = x.size() # [bs*n_ch, n_frame, n_dim]
        x = x.view(bs, max_inst_num, n_frame, n_dim) # [bs, n_ch, n_frame=375, n_dim]

        # x = x[:, :, :119999] # truncate so that spectrogram is 75Hz
        # x = torch.permute(self.transform(x), (0, 1, 3, 2)) # [bs, n_ch, n_frame=375, n_dim=128 (mel bins)]

        # Pad channels to maximum
        n_ch = x.size(1)
        tgt_ch = 10
        if n_ch < tgt_ch:
            x = torch.cat([x, torch.zeros(x.size(0), tgt_ch - n_ch, x.size(2), x.size(3), device=x.device)], dim=1)

        for conv in self.conv_blocks:
            x = conv(x)
        # [bs, n_ch_cnn, n_frame, n_dim_cnn_out]

        x = torch.permute(x, (0,2,1,3))  # [bs, n_frame, n_ch_cnn, n_dim_cnn_out]
        x = x.reshape(x.size(0), x.size(1), -1) # [bs, n_frame, cnn_dim = n_ch_cnn * n_dim_cnn_out = 4864]

        x, _ = self.rnn(x)

        x = x.contiguous()
        for fc in self.fc_blocks:
            x = fc(x)

        x = self.output_layer(x)
        return x


class MultiChannelMertAttnCRNN(nn.Module):
    '''
    CRNN with MERT features and channel attention
    '''
    def __init__(self, model_config=None):
        '''
        input_shape, output_dim, cnn_nb_filt, cnn_pool_size, rnn_nb, fc_nb, dropout_rate
        '''
        super().__init__()

        self.config = model_config
        self.mert = MERTModel.from_pretrained(model_config.mert_url)

        self.transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=24000,
            n_fft=2048,
            hop_length=320,
        )

        # There are 10 input channels at most (include mixture channel)

        input_shape = [1, 375, 2048] # [n_ch, n_frame, n_dim]
        output_dim = 16 + 1 # Same as my inst_cls model, n_inst + 1 for na
        cnn_nb_filt = 128 # From the repo
        cnn_pool_size = [5, 2, 2]
        rnn_nb = [32, 32]
        rnn_hidden_size = 32
        fc_nb = [32]
        dropout_rate = 0.

        self.conv_blocks = nn.ModuleList()
        self.rnn_blocks = nn.ModuleList()
        self.fc_blocks = nn.ModuleList()

        current_channels = input_shape[0]
        current_length = input_shape[1]

        # Convolutional layers
        for i, pool_size in enumerate(cnn_pool_size):
            self.conv_blocks.append(nn.Sequential(
                nn.Conv2d(current_channels, cnn_nb_filt, kernel_size=(3, 3), padding='same'),
                nn.BatchNorm2d(cnn_nb_filt),
                nn.ReLU(),
                nn.MaxPool2d((1, pool_size)),
                nn.Dropout(dropout_rate)
            ))
            current_channels = cnn_nb_filt  # update the channel number for the next layer

        # RNN layers
        self.rnn = nn.GRU(input_size=4864,
                            hidden_size=rnn_hidden_size, 
                            num_layers=2, 
                            batch_first=True, 
                            bidirectional=True, 
                            dropout=dropout_rate
                            )

        # Fully connected layers
        for f in fc_nb:
            self.fc_blocks.append(nn.Sequential(
                nn.Linear(rnn_hidden_size * 2, f),  # *2 because of bidirectional
                nn.Dropout(dropout_rate)
            ))
            r = f  # next input features will be the output of the current layer

        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(r, output_dim),
            nn.Sigmoid()
        )

        self.inst_emb = nn.Embedding(model_config.num_inst_type + 1, 768)
        self.channel_attn = nn.MultiheadAttention(embed_dim=768, num_heads=model_config.attn_heads, batch_first=True)
        self.attn_drop = nn.Dropout(model_config.attn_drop) # zero out attn_drop% of the attention weights

    def forward(self, x, insts=None):
        '''
        x: [bs, n_ch, n_frame, n_dim]
        '''

        (bs, n_ch, n_samples) = x.size()
        x = x.view(-1, x.size(-1))
        x = self.mert(x).last_hidden_state
        (_, n_frame, n_dim) = x.size() # [bs*n_ch, n_frame, n_dim]
        x = x.view(bs, n_ch, n_frame, n_dim) # [bs, n_ch, n_frame=375, n_dim]

        # Add Instrument Embedding
        inst_emb = self.inst_emb(insts)
        x = x + inst_emb.unsqueeze(2) # [bs, n_ch, n_frame, n_dim]

        # Prepare the mixture channel
        mix_channel_id = self.config['mix_ch_id']
        mix_idx = (insts == mix_channel_id).nonzero(as_tuple=True)[1]
        mix_x = x[torch.arange(bs), mix_idx, :, :].unsqueeze(1)
        mix_x = mix_x.view(bs * n_frame, n_dim).unsqueeze(1) # [bs*n_frame, 1, n_dim]

        # Prepare other channels as k and v
        x = x.permute(0, 2, 1, 3) # [bs, n_frame, n_ch, n_dim]
        x = x.reshape(bs * n_frame, n_ch, n_dim)    # [bs*n_frame, n_ch, n_dim]

        # Channel-wise attention
        (x, _) = self.channel_attn(mix_x, x, x) # [bs*n_frame, 1, n_dim]
        x = self.attn_drop(x)

        x = x.squeeze(1)    # [bs*n_frame, n_dim]
        x = x.view(bs, n_frame, n_dim)  # [bs, n_frame, n_dim]
        x = x.unsqueeze(1)

        for conv in self.conv_blocks:
            x = conv(x)

        x = torch.permute(x, (0,2,1,3))  # [bs, n_frame, n_ch_cnn, n_dim_cnn_out]
        x = x.reshape(x.size(0), x.size(1), -1) # [bs, n_frame, cnn_dim = n_ch_cnn * n_dim_cnn_out = 4864]

        x, _ = self.rnn(x)

        x = x.contiguous()
        for fc in self.fc_blocks:
            x = fc(x)

        x = self.output_layer(x)
        return x