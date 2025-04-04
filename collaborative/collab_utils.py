from scipy.sparse import csr_matrix
from pandas import DataFrame
import pandas as pd
import numpy as np
import re

def read_impressions_tsv(path="./MIND/train/behaviors.tsv"):
  return pd.read_csv(
      path,
      sep="\t",
      header=None,
      names=["impressionId", "userId", "timestamp", "history", "impressions"],
      usecols=["userId", "impressions"]
    ) 

def _split_clicked(x):
  return re.findall(r"(\d+)-1", x)

def _str2int(x):
  m = re.search(r"\d+", x)
  if not m:
    raise Exception("didn't find news id")
  return int(m.group())

# ignore articles that have not been clicked
def preprocess_clicked(df: DataFrame, rows=100_000):
  df["newsId"] = df["impressions"].apply(_split_clicked)
  df = df.explode("newsId").reset_index()
  df = df.head(rows)

  df["userId"] = df["userId"].apply(_str2int)
  df["newsId"] = df["newsId"].apply(_str2int)
  df["click"] = 1

  return df[["userId", "newsId", "click"]]


# Create space efficient sparse matrix and index mappers
def create_x(df: DataFrame):
  user_ids = df["userId"].unique()
  news_ids = df["newsId"].unique()

  M = len(user_ids)
  N = len(news_ids)

  uid2index = {id: i for i, id in enumerate(user_ids)}
  nid2index = {id: i for i, id in enumerate(news_ids)}    # Create sparse matrix and mappers
    
  index2uid = {v: k for k, v in uid2index.items()}
  index2nid = {v: k for k, v in nid2index.items()}

  rows = [uid2index[uid] for uid in df["userId"]]
  cols = [nid2index[nid] for nid in df["newsId"]]

  data = df["click"]

  X = csr_matrix((data, (rows, cols)), shape=(M, N))

  return X, uid2index, nid2index, index2uid, index2nid