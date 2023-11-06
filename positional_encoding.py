import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_size, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, embedding_size)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, embedding_size, 2).float()
            * -(math.log(10000.0) / embedding_size)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # Add positional encoding to input
        x = x + self.pe[:, : x.size(1)]
        return x
