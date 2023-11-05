import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleHeadAttention(nn.Module):
    def __init__(self, embedding_dim):
        super(SingleHeadAttention, self).__init__()
        self.embedding_dim = embedding_dim
        
        # Linear layers to compute Q, K, V
        self.query_layer = nn.Linear(embedding_dim, embedding_dim)
        self.key_layer = nn.Linear(embedding_dim, embedding_dim)
        self.value_layer = nn.Linear(embedding_dim, embedding_dim)
        
        # Output linear layer
        self.output_layer = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        # x is of shape (batch_size, seq_length, embedding_dim)
        Q = self.query_layer(x)
        K = self.key_layer(x)
        V = self.value_layer(x)
        
        # Calculate attention scores: (batch_size, seq_length, seq_length)
        attention_scores = torch.matmul(Q, K.transpose(2, 1)) / self.embedding_dim**0.5
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Compute the context vector
        context = torch.matmul(attention_weights, V)
        
        # Pass through the output layer
        output = self.output_layer(context)
        
        return output
