import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".." ))
)

from collaborative.collab_utils import create_x, read_impressions_tsv

class MatrixFactorizationRecommender:
    """
    A news recommendation system using matrix factorization.
    
    This class implements matrix factorization for collaborative filtering to recommend
    news articles to users based on historical interaction data.
    """
    
    def __init__(self, df, preprocessed=False, n_factors=10, n_iterations=20, learning_rate=0.01, 
                 regularization=0.1, random_state=None):
        """
        Initialize the recommender system.
        
        Parameters:
            n_factors (int): Number of latent factors
            n_iterations (int): Number of training iterations
            learning_rate (float): Learning rate for gradient descent
            regularization (float): Regularization parameter to prevent overfitting
            random_state (int): Random seed for reproducibility
        """
        if not preprocessed:
          df = df.dropna()
          df = df.copy()
          df["news_id"] = df["history"].apply(lambda x: x.split())
          df = df.explode("news_id").reset_index()
          df["click"] = 1
          df = df[["user_id", "news_id", "click"]]

        self.df = df
        self.n_factors = n_factors
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.regularization = regularization
        
        # Set random seed if provided
        if random_state is not None:
            np.random.seed(random_state)
            
        # These will be set during training
        self.user_factors = None
        self.item_factors = None
        self.uid2index = None
        self.nid2index = None
        self.index2uid = None
        self.index2nid = None
        self.user_biases = None
        self.item_biases = None
        self.global_bias = None
    
    def _sigmoid(self, x):
        """Apply sigmoid function to scale predictions between 0 and 1"""
        return 1 / (1 + np.exp(-x))
    
    def train(self):
        """
        Train the matrix factorization model.
        
        Parameters:
            df (pd.DataFrame): DataFrame with columns userId, newsId, click
            
        Returns:
            self: The trained model
        """
        df = self.df
        # Clean data - remove NaN values
        df = df.dropna(subset=["news_id"])
        
        # Create interaction matrix and mappings
        X, self.uid2index, self.nid2index, self.index2uid, self.index2nid = create_x(df)
        
        # Get dimensions
        n_users, n_items = X.shape
        
        # Initialize factors with random values
        self.user_factors = np.random.normal(
            scale=1.0 / self.n_factors, 
            size=(n_users, self.n_factors)
        )
        self.item_factors = np.random.normal(
            scale=1.0 / self.n_factors, 
            size=(n_items, self.n_factors)
        )
        
        # Initialize biases
        self.user_biases = np.zeros(n_users)
        self.item_biases = np.zeros(n_items)
        self.global_bias = X.data.mean()
        
        # Convert to dense for easier indexing during training
        X_dense = X.toarray()
        
        # Mask to identify non-zero elements
        mask = X_dense > 0
        
        # Stochastic Gradient Descent
        for iteration in range(self.n_iterations):
            # Compute predictions
            predicted = self._predict_matrix()
            
            # Compute errors only for observed interactions
            errors = np.zeros_like(X_dense)
            errors[mask] = X_dense[mask] - predicted[mask]
            
            # MSE for this iteration
            #mse = (errors[mask] ** 2).mean()
            #print(f"Iteration {iteration + 1}/{self.n_iterations}, MSE: {mse:.4f}")
            
            # Update factors and biases
            for u in range(n_users):
                # Get items this user has interacted with
                item_indices = np.where(mask[u])[0]
                if len(item_indices) == 0:
                    continue
                
                # Update user bias
                self.user_biases[u] += self.learning_rate * (
                    errors[u, item_indices].sum() - self.regularization * self.user_biases[u]
                )
                
                # Update user factors
                for f in range(self.n_factors):
                    user_factor_update = 0
                    for i in item_indices:
                        user_factor_update += errors[u, i] * self.item_factors[i, f]
                    
                    user_factor_update -= self.regularization * self.user_factors[u, f]
                    self.user_factors[u, f] += self.learning_rate * user_factor_update
            
            for i in range(n_items):
                # Get users who have interacted with this item
                user_indices = np.where(mask[:, i])[0]
                if len(user_indices) == 0:
                    continue
                
                # Update item bias
                self.item_biases[i] += self.learning_rate * (
                    errors[user_indices, i].sum() - self.regularization * self.item_biases[i]
                )
                
                # Update item factors
                for f in range(self.n_factors):
                    item_factor_update = 0
                    for u in user_indices:
                        item_factor_update += errors[u, i] * self.user_factors[u, f]
                    
                    item_factor_update -= self.regularization * self.item_factors[i, f]
                    self.item_factors[i, f] += self.learning_rate * item_factor_update
        
        return self
    
    def _predict_matrix(self):
        """
        Generate the complete prediction matrix.
        
        Returns:
            np.ndarray: Matrix of predicted ratings
        """
        # Dot product of user and item factors
        predictions = np.dot(self.user_factors, self.item_factors.T)
        
        # Add biases
        predictions += self.global_bias
        predictions += self.user_biases[:, np.newaxis]
        predictions += self.item_biases[np.newaxis, :]
        
        # Apply sigmoid to scale between 0 and 1
        return self._sigmoid(predictions)
    
    def predict(self, user_id, news_id):
        """
        Predict the interaction between a user and news item.
        
        Parameters:
            user_id (str): User ID
            news_id (str): News ID
            
        Returns:
            float: Predicted interaction score
        """
        if user_id not in self.uid2index or news_id not in self.nid2index:
            return self.global_bias
        
        u_idx = self.uid2index[user_id]
        i_idx = self.nid2index[news_id]
        
        # Compute prediction
        prediction = np.dot(self.user_factors[u_idx], self.item_factors[i_idx])
        prediction += self.global_bias + self.user_biases[u_idx] + self.item_biases[i_idx]
        
        return self._sigmoid(prediction)
    
    def user_exist(self, user_id):
        return user_id in self.uid2index
    
    def recommend(self, user_id, N=10):
        """
        Generate recommendations for a user.
        
        Parameters:
            user_id (str): User ID
            N (int): Number of recommendations to generate
            
        Returns:
            list: List of recommended news IDs
        """
        
        # Check if user exists in the model
        if user_id not in self.uid2index:
            print(f"User {user_id} not found in training data.")
            return []
        
        u_idx = self.uid2index[user_id]
        
        # Get all item predictions for this user
        user_vector = self.user_factors[u_idx].reshape(1, -1)
        predictions = np.dot(user_vector, self.item_factors.T).flatten()
        predictions += self.global_bias + self.user_biases[u_idx] + self.item_biases
        predictions = self._sigmoid(predictions)
        
        # Get indices of top N recommendations
        top_indices = np.argsort(predictions)[::-1][:N]
        
        # Convert indices to news IDs
        recommendations = [self.index2nid[idx] for idx in top_indices]
        
        return recommendations
  


# Example usage
def main():
    df = read_impressions_tsv()

    df = df.dropna()
    df = df.copy()
    df["news_id"] = df["history"].apply(lambda x: x.split())
    df = df.explode("news_id").reset_index()
    df = df.head(1000)
    df["click"] = 1
    df = df[["user_id", "news_id", "click"]]
    
    # Create and train recommender
    recommender = MatrixFactorizationRecommender(
        n_factors=10, 
        n_iterations=20,
        learning_rate=0.01,
        regularization=0.1
    )
    
    recommender.fit(df)
    
    # Get recommendations for a specific user
    user_id = 'U13740'
    recommendations = recommender.recommend(user_id, N=5)
    print(f"Recommendations for user {user_id}: {recommendations}")
    


if __name__ == "__main__":
    main()