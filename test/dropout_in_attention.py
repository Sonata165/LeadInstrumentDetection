'''
Test the effect of dropout in attention.

Author: Longshen Ou, 2024/07/25
'''

import os
import sys
import torch

def main():
    l = torch.nn.MultiheadAttention(embed_dim=32, num_heads=1, batch_first=True)
    l2 = torch.nn.MultiheadAttention(embed_dim=32, num_heads=1, batch_first=True, dropout=0.5)

    # Copy l's parameters to l2
    l2.load_state_dict(l.state_dict())

    inp = torch.randn(2, 5, 32)
    out1 = l(inp, inp, inp)
    out2 = l2(inp, inp, inp)

    a = 2

if __name__ == '__main__':
    main()