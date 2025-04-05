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
      names=["impressionId", "user_id", "timestamp", "history", "impressions"],
      usecols=["user_id", "history", "impressions"]
    ) 

def _split_clicked(x):
  return re.findall(r"(\w+)-1", x)


def preprocess_clicked(df: DataFrame, rows=100_000):
  df["news_id"] = df["impressions"].apply(_split_clicked)
  df = df.explode("news_id").reset_index()
  df = df.head(rows)

  df["click"] = 1

  return df[["user_id", "news_id", "click"]]

# ignore articles that have not been clicked
def preprocess_history(df: DataFrame, rows=100_000):

  df = df.dropna()

  df["news_id"] = df["history"].str.split()
  df = df.explode("news_id").reset_index()
  df = df.head(rows)

  df["click"] = 1

  return df[["user_id", "news_id", "click"]]


# Create space efficient sparse matrix and index mappers
def create_x(df: DataFrame):
  user_ids = df["user_id"].unique()
  news_ids = df["news_id"].unique()

  M = len(user_ids)
  N = len(news_ids)

  uid2index = {id: i for i, id in enumerate(user_ids)}
  nid2index = {id: i for i, id in enumerate(news_ids)}
    
  index2uid = {v: k for k, v in uid2index.items()}
  index2nid = {v: k for k, v in nid2index.items()}

  rows = [uid2index[uid] for uid in df["user_id"]]
  cols = [nid2index[nid] for nid in df["news_id"]]

  data = df["click"]

  X = csr_matrix((data, (rows, cols)), shape=(M, N))

  return X, uid2index, nid2index, index2uid, index2nid