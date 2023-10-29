import torch

def generate_text(model, idx_to_word, word_to_idx, initial_text="the", max_length=100):
    model.eval()  # Set the model to evaluation mode
    
    # Tokenize the initial text
    tokens = initial_text.split()
    input_sequence = [word_to_idx[word] for word in tokens]
    input_sequence = torch.tensor(input_sequence).view(1, -1)
    
    generated_text = tokens.copy()
    
    with torch.no_grad():
        for _ in range(max_length):
            # Forward pass
            output = model(input_sequence)
            
            # Get the last predicted token
            last_output = output[0, -1, :]
            
            # Sample a token from the output distribution
            _, sampled_token_idx = torch.max(last_output, dim=0)
            
            # Add the sampled token to the generated text
            sampled_token = idx_to_word[sampled_token_idx.item()]
            generated_text.append(sampled_token)
            
            # Update the input sequence for the next iteration
            input_sequence = torch.cat([input_sequence, sampled_token_idx.view(1, 1)], dim=1)
    
    return ' '.join(generated_text)
