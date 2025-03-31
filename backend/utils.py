import pandas as pd

def load_news_df():
  df = pd.read_csv(
    "MIND/news.tsv",
    sep="\t",
    header=None,
    names=['newsId', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities']

  )
  df = df.dropna()
  return df