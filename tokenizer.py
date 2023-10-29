from collections import Counter

def tokenize_text(text):
    # Tokenize the text into words
    tokens = text.split()

    # Count the occurrences of each token
    token_counts = Counter(tokens)

    # Create a vocabulary by sorting the unique tokens
    vocab = sorted(token_counts, key=token_counts.get, reverse=True)

    # Create word-to-index and index-to-word mappings
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}

    return vocab, word_to_idx, idx_to_word
