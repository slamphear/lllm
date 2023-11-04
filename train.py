from data_loader import load_sample_data
from tokenizer import tokenize_text
from model import SimpleLSTM
import torch
import torch.optim as optim

# Hyperparameters
embedding_dim = 50
hidden_dim = 128
num_layers = 2
batch_size = 64
seq_length = 30
num_epochs = 10

# Create batches
def create_batches(tokens, word_to_idx, batch_size, seq_length):
    # Convert tokens to integers
    tokens = [word_to_idx[word] for word in tokens]
    
    # Number of sequences in a batch
    num_seq = len(tokens) // seq_length
    
    # Trim tokens to make it fit evenly into batches
    tokens = tokens[:num_seq * seq_length]
    
    # Create input and target sequences
    input_data = torch.tensor(tokens).view(-1, seq_length)[:num_seq * batch_size]
    target_data = torch.tensor(tokens[1:] + [tokens[0]]).view(-1, seq_length)[:num_seq * batch_size]

    # Create batches
    num_batches = len(input_data) // batch_size
    input_batches = input_data[:num_batches * batch_size].view(-1, batch_size, seq_length)
    target_batches = target_data[:num_batches * batch_size].view(-1, batch_size, seq_length)
    
    return input_batches, target_batches

def train_model(model, vocab, input_batches, target_batches):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(num_epochs):
        print(f'Starting epoch {epoch+1}/{num_epochs}:\nBatch ', end='')
        for i in range(input_batches.size(0)):
            print(f'{i+1}/{input_batches.size(0)}...', end='')
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            output = model(input_batches[i])
            
            # Compute loss
            loss = criterion(output.view(-1, len(vocab)), target_batches[i].view(-1))
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
        print(f'Finished epoch {epoch+1}/{num_epochs}.')
