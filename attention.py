import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embedding_size, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        assert (
            embedding_size % num_heads == 0
        ), "Embedding size must be divisible by number of heads"

        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.head_dim = embedding_size // num_heads

        # Define weight matrices for Q, K, V for each head
        self.Wq = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.Wk = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.Wv = nn.Linear(self.head_dim, self.head_dim, bias=False)

        # Linear layer to transform concatenated outputs
        self.fc_out = nn.Linear(embedding_size, embedding_size)

    def forward(self, query, key, value):
        batch_size = query.shape[0]
        seq_length = query.shape[1]

        # Split embedding into heads
        query = query.view(batch_size, seq_length, self.num_heads, self.head_dim)
        key = key.view(batch_size, seq_length, self.num_heads, self.head_dim)
        value = value.view(batch_size, seq_length, self.num_heads, self.head_dim)

        # Apply linear layers
        query = self.Wq(query)
        key = self.Wk(key)
        value = self.Wv(value)

        # Scaled dot-product attention
        scaled_dot_product = torch.einsum("ijkl,ijml->ijkm", [query, key]) / (
            self.head_dim**0.5
        )
        attention_weights = F.softmax(scaled_dot_product, dim=-1)
        output = torch.einsum("ijkm,ijlm->ijkl", [attention_weights, value])

        # Concatenate and transform
        output = output.view(batch_size, seq_length, -1)
        output = self.fc_out(output)

        return output
