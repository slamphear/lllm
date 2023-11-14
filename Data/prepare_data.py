def prepare_data(sample_texts, word_to_idx, max_sample_length=2000):
    # Tokenize and truncate each article separately
    token_indices = []
    for text in sample_texts:
        tokens = text.split()
        tokens = tokens[:max_sample_length]
        token_indices.extend(
            [word_to_idx.get(word, word_to_idx["<UNK>"]) for word in tokens]
        )
        # Optional: add padding between articles if needed
        # token_indices += [word_to_idx['<PAD>']] * (max_sample_length - len(tokens))

    return token_indices
