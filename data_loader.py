import opendatasets as od
import pandas as pd


def load_sample_data(num_samples: int) -> list:
    dataset_url = "https://www.kaggle.com/datasets/jjinho/wikipedia-20230701"
    od.download(dataset_url)

    # Just sample num_samples rows from the letter A for now
    df = pd.read_parquet("wikipedia-20230701/a.parquet")
    sampled_texts = df.sample(n=num_samples)["text"].tolist()

    return sampled_texts
