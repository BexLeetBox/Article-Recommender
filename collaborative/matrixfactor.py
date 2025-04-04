
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from pandas import DataFrame

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".." ))
)


from collaborative.collab_utils import create_x
from utils.evaluation import evaluate_model



# Matrix Factorization class
class MatrixFactorization:
    def __init__(self, impressions: DataFrame, k=10, alpha=0.01, beta=0.01, iterations=20):
        self.skipped = 0
        """
        Initialize matrix factorization model
        
        Parameters:
        -----------
        X : scipy.sparse.csr_matrix
            User-item interaction matrix
        k : int
            Number of latent features
        alpha : float
            Learning rate
        beta : float
            Regularization parameter
        iterations : int
            Number of iterations to train the model
        """
        self.X, self.uid2index, self.nid2index, self.index2uid, self.index2nid = create_x(impressions)
        self.k = k
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations
        self.n_users, self.n_items = self.X.shape
        
        # Initialize user and item latent feature matrices
        self.P = np.random.normal(scale=1/self.k, size=(self.n_users, self.k))
        self.Q = np.random.normal(scale=1/self.k, size=(self.n_items, self.k))
        
        # Get indices of non-zero elements
        self.rows, self.cols = self.X.nonzero()
        self.n_ratings = len(self.rows)
        
    def train(self):
        """Train model with stochastic gradient descent"""
        for _ in range(self.iterations):
            self._sgd()
        
    def _sgd(self):
        """Perform stochastic gradient descent"""
        for i in range(self.n_ratings):
            u = self.rows[i]
            i = self.cols[i]
            prediction = self.predict(u, i)
            e = self.X[u, i] - prediction
            
            # Update matrices
            self.P[u] += self.alpha * (e * self.Q[i] - self.beta * self.P[u])
            self.Q[i] += self.alpha * (e * self.P[u] - self.beta * self.Q[i])
            
    def predict(self, u, i):
        """Predict rating of item i by user u"""
        return self.P[u].dot(self.Q[i].T)

    def recommend(self, user_id, N=10, exclude_clicked=True):
        """
        Recommend top N items for a user
        
        Parameters:
        -----------
        user_idx : int
            Internal user index
        N : int
            Number of recommendations
        exclude_clicked : bool
            Whether to exclude items that the user has already clicked
            
        Returns:
        --------
        list
            List of tuples (item_idx, predicted_score)
        """
        try:
            user_idx = self.uid2index[user_id]
        except:
            self.skipped += 1
            return []
        predictions = self.P[user_idx].dot(self.Q.T)
        
        if exclude_clicked:
            # Get indices of items that user has already interacted with
            clicked_items = self.X[user_idx].indices
            
            # Set predictions for those items to negative infinity
            predictions[clicked_items] = float('-inf')
        
        # Get indices of top N items
        top_n_idx = np.argsort(-predictions)[:N]

        recommended = [self.index2nid[i] for i in top_n_idx]
        
        return recommended


raw_impressions = pd.read_csv(
    "./MIND/train/behaviors.tsv",
    sep='\t',
    header=None,
    names=["impressionId", "userId", "time", "history", "impressions"],
    usecols=["userId", "history", "impressions"]
  )

import re

impressions = raw_impressions

def split_clicked(x):
  m = re.findall(r"(\w+)-1", x)
  if not m:
    raise Exception("didn't find news id")
  return m



def combine_interractions(x):
  history = x["history"].split(" ")
  clicked = re.findall(r"(\w+)-1", x["impressions"])
  return history + clicked


impressions = impressions.dropna()

impressions["newsId"] = impressions.apply(combine_interractions, axis=1)
impressions = impressions.explode("newsId").reset_index()
impressions = impressions.head(100_000)

raw_impressions = raw_impressions[raw_impressions["userId"].isin(impressions["userId"])]

print(raw_impressions.info())


impressions["click"] = 1

impressions = impressions[["userId", "newsId", "click"]]

print("start training...")

model = MatrixFactorization(impressions)
model.train()

print("done")


print("start evaluating...")
avg_ndcg, avg_auc, avg_mrr = evaluate_model(model, raw_impressions)
print("done")

print(f"skipped {model.skipped}")
print(f"avg_ndcg {avg_ndcg}")
print(f"avg_auc {avg_auc}")
print(f"avg_mrr {avg_mrr}")

""" 
10_000 entries
avg_ndcg 0.0024746115538230684
avg_auc 0.0025963650888755742
avg_mrr 0.002482239150903021

100_000 entries
avg_ndcg 0.000980072209624899
avg_auc 0.0010293901388626291
avg_mrr 0.0007336530545076584
"""