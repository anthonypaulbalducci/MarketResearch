import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device {device}.")

def make_causal_mask(seq_len):
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask.bool().to(device)

def make_attention_mask(seq_len):
    mask = torch.ones(seq_len, seq_len)
    mask = mask.masked_fill(mask == 0, float('-inf'))
    return mask.to(device)

class MarketMaker(nn.Module):
    def __init__(self, seq_len, n_features, n_hidden, n_layers, n_heads, dropout=0.1):
        super(MarketMaker, self).__init__()