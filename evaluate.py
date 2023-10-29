import torch
import math

def evaluate(model, input_batches, target_batches, criterion, vocab_size):
    model.eval()  # Set the model to evaluation mode
    
    total_loss = 0
    with torch.no_grad():
        for i in range(input_batches.size(0)):
            # Forward pass
            output = model(input_batches[i])

            # Compute loss
            loss = criterion(output.view(-1, vocab_size), target_batches[i].view(-1))
            total_loss += loss.item()
        
    # Calculate perplexity
    perplexity = math.exp(total_loss / input_batches.size(0))
    
    return perplexity
