import torch
import torch.nn as nn

class SimpleLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(SimpleLSTM, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x):
        # Pass data through embedding layer
        x = self.embedding(x)
        
        # Pass embeddings through LSTM
        lstm_out, _ = self.lstm(x)
        
        # Pass the LSTM output through the fully connected layer
        output = self.fc(lstm_out)
        
        return output
