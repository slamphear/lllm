from data_loader import load_sample_data
from tokenizer import tokenize_text
from model import SimpleLSTM
from train import create_batches, train_model
from evaluate import evaluate
from inference import generate_text
import torch

def main():
    # 1. Load and tokenize data
    sample_text = load_sample_data('sample_data.txt')
    vocab, word_to_idx, idx_to_word = tokenize_text(sample_text)
    tokens = sample_text.split()

    # 2. Create batches
    input_batches, target_batches = create_batches(tokens, word_to_idx, batch_size=3, seq_length=30)

    # 3. Initialize model
    model = SimpleLSTM(vocab_size=len(vocab), embedding_dim=50, hidden_dim=128, num_layers=2)

    # 4. Train the model
    train_model(model, input_batches, target_batches)

    # 5. Evaluate the model
    perplexity = evaluate(model, input_batches, target_batches, criterion=torch.nn.CrossEntropyLoss(), vocab_size=len(vocab))
    print(f'Perplexity: {perplexity}')

    # 6. Generate text
    generated_text = generate_text(model, idx_to_word, word_to_idx, initial_text="alice", max_length=50)
    print(f'Generated Text: {generated_text}')

if __name__ == "__main__":
    main()
