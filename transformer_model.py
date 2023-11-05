import torch.nn as nn
from transformer_block import TransformerBlock


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, ff_hidden_dim, num_blocks):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Stack multiple transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embedding_dim, ff_hidden_dim) for _ in range(num_blocks)
        ])
        
        self.output_layer = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        # x is of shape (batch_size, seq_length)
        x = self.embedding(x)
        
        # Pass through each transformer block
        for block in self.transformer_blocks:
            x = block(x)
            
        # Output layer
        logits = self.output_layer(x)
        
        return logits
