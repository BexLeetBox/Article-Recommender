# data.py - Module for data loading with caching
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
from collections import Counter
from tempfile import TemporaryDirectory
import functools

from recommenders.datasets.mind import (download_mind)
from recommenders.datasets.download_utils import unzip_file

print("System version: {}".format(sys.version))

@functools.lru_cache(maxsize=1)
def get_news_data():
    print("Loading MIND dataset (this should only appear once)...")
    
    # MIND sizes: "demo", "small" or "large"
    mind_type = "demo"

    tmpdir = TemporaryDirectory()
    data_path = tmpdir.name
    train_zip, valid_zip = download_mind(size=mind_type, dest_path=data_path)
    unzip_file(train_zip, os.path.join(data_path, "train"), clean_zip_file=False)
    unzip_file(valid_zip, os.path.join(data_path, "valid"), clean_zip_file=False)
    output_path = os.path.join(data_path, "utils")
    os.makedirs(output_path, exist_ok=True)

    news = pd.read_table(
        os.path.join(data_path, "train", "news.tsv"),
        names=[
            "newid",
            "vertical",
            "subvertical",
            "title",
            "abstract",
            "url",
            "entities in title",
            "entities in abstract",
        ],
        usecols=["vertical", "subvertical", "title", "abstract"],
    )
    
    # To prevent the TemporaryDirectory from being deleted when the function returns
    # We need to save it as an attribute of the function
    get_news_data._tmpdir = tmpdir
    
    return news

news_data = None  