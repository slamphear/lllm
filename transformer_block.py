import torch.nn as nn
from attention import SingleHeadAttention


class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, ff_hidden_dim):
        super(TransformerBlock, self).__init__()
        self.attention = SingleHeadAttention(embedding_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(embedding_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, embedding_dim)
        )
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        # Apply attention
        attention_out = self.attention(x)
        x = self.norm1(x + attention_out)
        
        # Apply feedforward network
        ff_out = self.feedforward(x)
        x = self.norm2(x + ff_out)
        
        return x
