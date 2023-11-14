from collections import Counter


def tokenize_text(sample_texts):
    # Initialize a list to store all tokens
    all_tokens = []

    # Tokenize each text separately and extend the all_tokens list
    for text in sample_texts:
        tokens = text.split()
        all_tokens.extend(tokens)

    # Count the occurrences of each token
    token_counts = Counter(all_tokens)

    # Create a vocabulary by sorting the unique tokens
    vocab = sorted(token_counts, key=token_counts.get, reverse=True)

    # Create word-to-index and index-to-word mappings
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}

    return vocab, word_to_idx, idx_to_word
