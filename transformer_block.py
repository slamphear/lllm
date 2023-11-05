import torch.nn as nn

from attention import MultiHeadSelfAttention


class TransformerBlock(nn.Module):
    def __init__(self, embedding_size, num_heads, ff_hidden_factor):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadSelfAttention(embedding_size, num_heads)
        self.feedforward = nn.Sequential(
            nn.Linear(embedding_size, ff_hidden_factor * embedding_size),
            nn.ReLU(),
            nn.Linear(ff_hidden_factor * embedding_size, embedding_size),
        )
        self.norm1 = nn.LayerNorm(embedding_size)
        self.norm2 = nn.LayerNorm(embedding_size)

    def forward(self, x):
        # Apply attention
        attention_out = self.attention(x)
        x = self.norm1(x + attention_out)

        # Apply feedforward network
        ff_out = self.feedforward(x)
        x = self.norm2(x + ff_out)

        return x
