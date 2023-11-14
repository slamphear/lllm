import os
from collections import Counter
from configparser import ConfigParser

import torch

from Data.data_loader import load_sample_data
from Data.prepare_data import prepare_data
from Data.tokenizer import tokenize_text
from Model.evaluate import evaluate
from Model.inference import generate_text
from Model.train import create_batches, train_model
from Model.transformer_model import TransformerModel


def get_hyperparameters():
    config = ConfigParser()
    config.read("config.ini")

    hyperparameters = config["Hyperparameters"]
    num_samples = int(hyperparameters["num_samples"])
    max_sample_length = int(hyperparameters["max_sample_length"])
    batch_size = int(hyperparameters["batch_size"])
    seq_length = int(hyperparameters["seq_length"])
    num_epochs = int(hyperparameters["num_epochs"])
    learning_rate = float(hyperparameters["learning_rate"])
    scheduler_patience = int(hyperparameters["scheduler_patience"])
    scheduler_factor = float(hyperparameters["scheduler_factor"])
    max_vocab_size = int(hyperparameters["max_vocab_size"])
    embedding_size = int(hyperparameters["embedding_size"])
    num_heads = int(hyperparameters["num_heads"])
    ff_hidden_factor = int(hyperparameters["ff_hidden_factor"])
    num_blocks = int(hyperparameters["num_blocks"])
    initial_text = hyperparameters["initial_text"]
    max_len = int(hyperparameters["max_len"])
    temperature = float(hyperparameters["temperature"])

    return (
        num_samples,
        max_sample_length,
        batch_size,
        seq_length,
        num_epochs,
        learning_rate,
        scheduler_patience,
        scheduler_factor,
        max_vocab_size,
        embedding_size,
        num_heads,
        ff_hidden_factor,
        num_blocks,
        initial_text,
        max_len,
        temperature,
    )


def main():
    # 1. Load hyperparameters
    (
        num_samples,
        max_sample_length,
        batch_size,
        seq_length,
        num_epochs,
        learning_rate,
        scheduler_patience,
        scheduler_factor,
        max_vocab_size,
        embedding_size,
        num_heads,
        ff_hidden_factor,
        num_blocks,
        initial_text,
        max_len,
        temperature,
    ) = get_hyperparameters()

    # 2. Load and pre-process data
    print("Loading sample data...")
    sample_texts = load_sample_data(num_samples=num_samples)

    # Tokenize and build vocab
    vocab, word_to_idx, idx_to_word = tokenize_text(sample_texts)

    # Tokenize and build vocab
    tokens = [token for text in sample_texts for token in text.split()]

    if len(tokens) > max_vocab_size:
        # Count the frequency of each word in your corpus
        word_freqs = Counter(tokens)

        # Get the most common words up to MAX_VOCAB_SIZE
        # (Subtract 2 from max_vocab_size for <UNK> and <PAD>)
        vocab = [word for word, freq in word_freqs.most_common(max_vocab_size - 2)]

        # Add special tokens to vocab and word_to_idx
        vocab.extend(["<UNK>", "<PAD>"])
        word_to_idx = {word: idx for idx, word in enumerate(vocab)}

        # Replace all words not in the vocabulary with <UNK>
        tokens = [word if word in word_to_idx else "<UNK>" for word in tokens]

    # Prepare data: Truncate and pad sequences
    prepared_data = prepare_data(
        sample_texts, word_to_idx, max_sample_length=max_sample_length
    )

    # 3. Create batches
    if len(prepared_data) < batch_size * seq_length:
        print(
            f"Not enough data to create a batch. Data length: {len(prepared_data)} Batch size * Seq length: {batch_size * seq_length}"
        )
    else:
        print("Creating batches...")
        input_batches, target_batches = create_batches(
            prepared_data, batch_size, seq_length
        )

    # 4. Initialize or load model
    model_path = "model.pth"
    if os.path.exists(model_path):
        print("Loading existing model...")
        model = torch.load(model_path)
    else:
        print("Initializing new model...")
        model = TransformerModel(
            vocab_size=max_vocab_size,
            embedding_size=embedding_size,
            num_heads=num_heads,
            num_blocks=num_blocks,
            ff_hidden_factor=ff_hidden_factor,
            max_len=max_len,
        )

    # 5. Train the model
    print("Training model...")
    train_model(
        model,
        vocab,
        num_epochs,
        learning_rate,
        scheduler_patience,
        scheduler_factor,
        input_batches,
        target_batches,
    )

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
        max_len=max_len,
        temperature=temperature,
    )
    print(f"Generated Text: {generated_text}")


if __name__ == "__main__":
    main()
