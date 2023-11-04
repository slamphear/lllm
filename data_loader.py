import opendatasets as od
import pandas as pd


def load_sample_data() -> str:
    dataset_url = "https://www.kaggle.com/datasets/jjinho/wikipedia-20230701"
    od.download(dataset_url)

    # Just grab the first 100 rows from the letter A for now
    df = pd.read_parquet("wikipedia-20230701/a.parquet")
    sample_text = " ".join(df.head(100)["text"].values)

    return sample_text
