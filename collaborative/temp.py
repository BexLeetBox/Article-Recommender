import sys
import os

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".." ))
)

from collaborative.matrixfactor import MatrixFactorization
from collaborative.utils import create_x, preprocess_clicked, read_impressions_tsv


# Function to split data into train and test sets
def train_test_split_matrix(X, test_size=0.2):
    """Split the rating matrix into training and test sets"""
    # Get indices of non-zero elements
    rows, cols = X.nonzero()
    
    # Split indices
    train_indices, test_indices = train_test_split(
        range(len(rows)), test_size=test_size, random_state=42
    )
    
    # Create train and test matrices
    X_train = X.copy()
    X_test = csr_matrix(X.shape)
    
    # Remove test entries from train
    for i in test_indices:
        X_train[rows[i], cols[i]] = 0
    
    # Add test entries to test
    for i in test_indices:
        X_test[rows[i], cols[i]] = X[rows[i], cols[i]]
    
    # Recompute the nonzero elements
    X_train.eliminate_zeros()
    X_test.eliminate_zeros()
    
    return X_train, X_test

# Main function to run the recommender system
def run_news_recommender(impressions_df, k=10, alpha=0.01, beta=0.01, iterations=20):
    """
    Run the news recommender system
    
    Parameters:
    -----------
    impressions_df : pandas.DataFrame
        DataFrame with columns userId, newsId, click
    k : int
        Number of latent features
    alpha : float
        Learning rate
    beta : float
        Regularization parameter
    iterations : int
        Number of iterations
    
    Returns:
    --------
    model : MatrixFactorization
        Trained model
    uid2index : dict
        Mapping from user IDs to internal indices
    nid2index : dict
        Mapping from news IDs to internal indices
    """
    # Create sparse matrix and mappers
    X, uid2index, nid2index, index2uid, index2nid = create_x(impressions_df)
        
    # Split data
    X_train, X_test = train_test_split_matrix(X)
    
    # Create and train model
    model = MatrixFactorization(X_train, k=k, alpha=alpha, beta=beta, iterations=iterations)
    model.train()
    
    # Calculate RMSE on test data
    test_rmse = model.rmse()
    print(f"Test RMSE: {test_rmse}")
    
    return model, uid2index, nid2index, index2uid, index2nid

# Function to get recommendations for a specific user
def get_recommendations_for_user(user_id, model, uid2index, nid2index, index2nid, n=10):
    """
    Get top N recommendations for a specific user
    
    Parameters:
    -----------
    user_id : int
        External user ID
    model : MatrixFactorization
        Trained model
    uid2index : dict
        Mapping from user IDs to internal indices
    nid2index : dict
        Mapping from news IDs to internal indices
    index2nid : dict
        Mapping from internal indices to news IDs
    n : int
        Number of recommendations
        
    Returns:
    --------
    list
        List of tuples (newsId, predicted_score)
    """
    if user_id not in uid2index:
        print(f"User {user_id} not found in training data")
        return []
    
    user_idx = uid2index[user_id]
    top_n = model.recommend_top_n(user_idx, n=n)
    
    # Convert internal indices to external IDs
    recommendations = [(index2nid[item_idx], score) for item_idx, score in top_n]
    
    return recommendations


if __name__ == "__main__":
    impressions = read_impressions_tsv()
    impressions = preprocess_clicked(impressions)

    print(impressions.info())
    
    # Run the recommender system
    model, uid2index, nid2index, index2uid, index2nid = run_news_recommender(impressions)
    
    # Get recommendations for a user
    user_id = impressions["userId"].iloc[0]  # Example: first user in the dataset
    recommendations = get_recommendations_for_user(user_id, model, uid2index, nid2index, index2nid)
    
    print(f"Top recommendations for user {user_id}:")
    for news_id, score in recommendations:
        print(f"News ID: {news_id}, Score: {score:.4f}")
