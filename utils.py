'''
Various utility functions
Super useful

Author: Longshen Ou, 2024/07/25
'''

import os
import re
import json
import yaml
import torch
from bisect import bisect
from datetime import timedelta

jpath = os.path.join


def ls(dir_path, sort=True):
    '''
    A better version of os.path.join
    '''
    fns = os.listdir(dir_path)
    fns = [fn for fn in fns if not fn.startswith('.')]
    
    if sort:
        fns.sort()

    return fns


def read_json(path):
    with open(path, 'r', encoding='utf8') as f:
        data = f.read()
        data = json.loads(data)
    return data


def save_json(data, path, sort=False):
    with open(path, 'w', encoding='utf8') as f:
        f.write(json.dumps(data, indent=4, sort_keys=sort, ensure_ascii=False))


def print_json(data):
    print(json.dumps(data,indent=4,ensure_ascii=False))


def create_if_not_exist(fp):
    if os.path.exists(fp) is not True:
        os.makedirs(fp)


def timecode_to_seconds(timecode):
    """Convert a time string formatted as 'MM:SS.MS' to seconds."""
    minutes, seconds = timecode.split(':')
    total_seconds = int(minutes) * 60 + float(seconds)
    return total_seconds


def timecode_to_timedelta(timecode):
    '''
    Convert timecode 'MM:SS.XXX' to timedelta object
    '''
    m, s = timecode.strip().split(':')
    m = int(m)
    s = float(s)
    ret = timedelta(minutes=m, seconds=s)
    return ret


def timedelta_to_timecode(delta):
    time_str = str(delta)
    t = time_str.split('.')[-1]
    if len(t) == 6:
        time_code = time_str[:-3]
    else:
        time_code = '{}.000'.format(time_str)

    t = time_code.split(':')
    if len(t) == 3:
        time_code = ':'.join(t[1:])

    return time_code


def timecode_to_millisecond(timecode):
    m, s = timecode.split(':')
    s, ms = s.split('.')
    m = int(m)
    s = int(s)
    ms = int(ms)
    s += m * 60
    ms += s * 1000
    return ms


def convert_waveform_to_mono(waveform):
    '''
    Convert stereo waveform to mono waveform
    '''
    if waveform.shape[0] == 2:
        ret = torch.mean(waveform, dim=0)
    else:
        ret = waveform
    if ret.dim() != 2:
        ret = ret.unsqueeze(0)
    return ret


def get_latest_checkpoint(base_dir):
    # 构建lightning_logs的路径
    logs_dir = os.path.join(base_dir, 'lightning_logs')
    
    # 确保该目录存在
    if not os.path.exists(logs_dir):
        raise FileNotFoundError(f"The directory {logs_dir} does not exist.")

    # 查找所有的version_X文件夹，获取最大的版本号
    versions = [d for d in os.listdir(logs_dir) if re.match(r'^version_\d+$', d)]
    if not versions:
        raise ValueError("No version directories found in lightning_logs.")
    
    # 获取最大的版本号
    latest_version = max(versions, key=lambda v: int(v.split('_')[1]))
    latest_version_dir = os.path.join(logs_dir, latest_version, 'checkpoints')

    # 确保checkpoints目录存在
    if not os.path.exists(latest_version_dir):
        raise FileNotFoundError(f"No checkpoints directory found in {latest_version_dir}")

    # 检查checkpoints目录下的文件
    checkpoints = os.listdir(latest_version_dir)
    if len(checkpoints) != 1:
        raise AssertionError("There should be exactly one checkpoint file in the directory.")
    
    # 获取checkpoint文件的完整路径
    checkpoint_path = os.path.join(latest_version_dir, checkpoints[0])
    return latest_version, checkpoint_path


def check_model_param(model):
    '''
    Check if the model has the parameter
    '''
    pytorch_total_params=sum(p.numel() for p in model.parameters())
    pytorch_train_params=sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Totalparams:',format(pytorch_total_params,','))
    print('Trainableparams:',format(pytorch_train_params,','))


def update_dict_cnt(dic, key):
    if key not in dic:
        dic[key] = 0
    dic[key] += 1


class float_dict:
    def __init__(self, keys, values):
        '''
        NOTE: keys need to be sorted before building a float_dict

        - len(key) = len(values) or len(values) + 1
        - If X >= keys[i], value[i+1] will be assigned
        '''
        # assert len(values) == len(keys) + 1
        self.keys = keys
        self.values = values
    
    @classmethod
    def from_dict(cls, dic):
        keys = [float(i) for i in dic.keys()]
        values = [dic[key] for key in dic]
        t = float_dict(keys, values)
        return t

    def __getitem__(self, key):
        return self.category(key, self.keys, self.values)

    def category(self, fl, breakpoints, cat):
        i = bisect(breakpoints, fl)
        return cat[i]

	
def read_yaml(fp):
    with open(fp, 'r') as f:
        data = yaml.safe_load(f)
    return data


def save_yaml(data, fp):
    with open(fp, 'w') as f:
        yaml.safe_dump(data, f, sort_keys=False)


def print_yaml(data):
    print(yaml.safe_dump(data))


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