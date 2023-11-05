import torch.nn as nn

from transformer_block import TransformerBlock


class TransformerModel(nn.Module):
    def __init__(
        self, vocab_size, embedding_size, num_heads, num_blocks, ff_hidden_factor
    ):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size, num_heads)

        # Stack multiple transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(embedding_size, num_heads, ff_hidden_factor)
                for _ in range(num_blocks)
            ]
        )

        self.output_layer = nn.Linear(embedding_size, vocab_size)

    def forward(self, x, mask=None):
        # x is of shape (batch_size, seq_length)
        x = self.embedding(x)

        # Pass through each transformer block
        for block in self.transformer_blocks:
            x = block(x, mask)

        # Output layer
        logits = self.output_layer(x)

        return logits
