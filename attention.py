import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert embedding_size % num_heads == 0

        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.head_dim = embedding_size // num_heads

        # Define the linear transformations for Q, K, and V
        self.q_linear = nn.Linear(embedding_size, embedding_size)
        self.k_linear = nn.Linear(embedding_size, embedding_size)
        self.v_linear = nn.Linear(embedding_size, embedding_size)

        # Linear transformation for the output
        self.fc_out = nn.Linear(embedding_size, embedding_size)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).item()

    def forward(self, query, key, value, mask=None):
        N = query.shape[0]  # Batch size
        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)

        # Split embedding_size into num_heads different pieces
        Q = Q.view(N, -1, self.num_heads, self.head_dim)
        K = K.view(N, -1, self.num_heads, self.head_dim)
        V = V.view(N, -1, self.num_heads, self.head_dim)

        # Transpose for attention computation
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Calculate attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # Apply mask if provided
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e10)

        # Apply softmax to get attention weights
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # Calculate the new value vectors
        weighted = torch.matmul(attn_weights, V)

        # Reshape and concatenate heads
        weighted = weighted.transpose(1, 2).contiguous()
        concat = weighted.view(N, -1, self.embedding_size)

        # Pass through the final linear layer
        output = self.fc_out(concat)

        return output
