import opendatasets as od
import pandas as pd


def load_sample_data(number_of_samples: int) -> str:
    dataset_url = "https://www.kaggle.com/datasets/jjinho/wikipedia-20230701"
    od.download(dataset_url)

    # Just grab the first 100 rows from the letter A for now
    df = pd.read_parquet("wikipedia-20230701/a.parquet")
    sample_text = " ".join(df.sample(n=number_of_samples)["text"].values)

    return sample_text
