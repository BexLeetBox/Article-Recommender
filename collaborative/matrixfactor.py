
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

class MatrixFactorization:
    def __init__(self, X: csr_matrix, k=10, alpha=0.01, beta=0.01, iterations=20):
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
        self.X = X
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
    
    def predict_all(self):
        """Predict all ratings"""
        return self.P.dot(self.Q.T)
    
    def rmse(self):
        """Calculate RMSE"""
        xs, ys = self.X.nonzero()
        predictions = self.predict_all()
        y_true = np.array([self.X[x, y] for x, y in zip(xs, ys)])
        y_pred = np.array([predictions[x, y] for x, y in zip(xs, ys)])
        return sqrt(mean_squared_error(y_true, y_pred))
    
    def recommend_top_n(self, user_idx, n=10, exclude_clicked=True):
        """
        Recommend top N items for a user
        
        Parameters:
        -----------
        user_idx : int
            Internal user index
        n : int
            Number of recommendations
        exclude_clicked : bool
            Whether to exclude items that the user has already clicked
            
        Returns:
        --------
        list
            List of tuples (item_idx, predicted_score)
        """
        predictions = self.P[user_idx].dot(self.Q.T)
        
        if exclude_clicked:
            # Get indices of items that user has already interacted with
            clicked_items = self.X[user_idx].indices
            
            # Set predictions for those items to negative infinity
            predictions[clicked_items] = float('-inf')
        
        # Get indices of top N items
        top_n_idx = np.argsort(-predictions)[:n]
        
        # Return item indices with predicted scores
        return [(idx, predictions[idx]) for idx in top_n_idx]
