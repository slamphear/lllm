def load_sample_data(file_path):
    with open(file_path, 'r') as f:
        return f.read().lower()
