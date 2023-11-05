import os
from collections import Counter
from configparser import ConfigParser

import torch

from data_loader import load_sample_data
from evaluate import evaluate
from inference import generate_text
from model import SimpleLSTM
from tokenizer import tokenize_text
from train import create_batches, train_model


def get_hyperparameters():
    config = ConfigParser()
    config.read("config.ini")

    hyperparameters = config["Hyperparameters"]
    number_of_samples = int(hyperparameters["number_of_samples"])
    batch_size = int(hyperparameters["batch_size"])
    seq_length = int(hyperparameters["seq_length"])
    max_vocab_size = int(hyperparameters["max_vocab_size"])
    embedding_dim = int(hyperparameters["embedding_dim"])
    hidden_dim = int(hyperparameters["hidden_dim"])
    num_layers = int(hyperparameters["num_layers"])
    initial_text = hyperparameters["initial_text"]
    max_length = int(hyperparameters["max_length"])
    temperature = float(hyperparameters["temperature"])

    return (
        number_of_samples,
        batch_size,
        seq_length,
        max_vocab_size,
        embedding_dim,
        hidden_dim,
        num_layers,
        initial_text,
        max_length,
        temperature,
    )


def main():
    # 1. Load hyperparameters
    (
        number_of_samples,
        batch_size,
        seq_length,
        max_vocab_size,
        embedding_dim,
        hidden_dim,
        num_layers,
        initial_text,
        max_length,
        temperature,
    ) = get_hyperparameters()

    # 2. Load and tokenize data
    print("Loading sample data...")
    sample_text = load_sample_data(number_of_samples=number_of_samples)
    vocab, word_to_idx, idx_to_word = tokenize_text(sample_text)
    tokens = sample_text.split()

    if len(tokens) > max_vocab_size:
        # Count the frequency of each word in your corpus
        word_freqs = Counter(tokens)

        # Get the most common words up to MAX_VOCAB_SIZE
        vocab = [word for word, freq in word_freqs.most_common(max_vocab_size-1)]

        # Add the special <UNK> token to the vocabulary
        vocab.append("<UNK>")

        # Create word_to_idx dictionary
        word_to_idx = {word: idx for idx, word in enumerate(vocab)}

        # Replace all words not in the vocabulary with <UNK>
        tokens = [word if word in word_to_idx else "<UNK>" for word in tokens]

    # 3. Create batches
    print("Creating batches...")
    input_batches, target_batches = create_batches(
        tokens, word_to_idx, batch_size=batch_size, seq_length=seq_length
    )

    # 4. Initialize or load model
    model_path = "model.pth"
    if os.path.exists(model_path):
        print("Loading existing model...")
        model = torch.load(model_path)
    else:
        print("Initializing new model...")
        model = SimpleLSTM(
            vocab_size=max_vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )

    # 5. Train the model
    print("Training model...")
    train_model(model, vocab, input_batches, target_batches)

    # Save the trained model
    print("Saving model...")
    torch.save(model, model_path)

    # 6. Evaluate the model
    perplexity = evaluate(
        model,
        input_batches,
        target_batches,
        criterion=torch.nn.CrossEntropyLoss(),
        vocab_size=max_vocab_size,
    )
    print(f"Perplexity: {perplexity}")

    # 7. Generate text
    generated_text = generate_text(
        model,
        idx_to_word,
        word_to_idx,
        initial_text=initial_text,
        max_length=max_length,
        temperature=temperature,
    )
    print(f"Generated Text: {generated_text}")


if __name__ == "__main__":
    main()
