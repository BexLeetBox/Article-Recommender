from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import sys
import os

from sklearn.neighbors import NearestNeighbors


sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".." ))
)

from collaborative.collab_utils import create_x, preprocess_clicked, read_impressions_tsv

class ItemSimilarityRecommender:
  
  def user_exist(self, user_id):
     return user_id in self.uid2index

  def __init__(self, df=None, preprocessed=False):
    if df is None:
      df = read_impressions_tsv("../MIND/train/behaviors.tsv")

    if not preprocessed:
       df = preprocess_clicked(df)

    self.df = df


  def train(self):
    self.X, self.uid2index, _, _, self.index2nid = create_x(self.df)
    item_knn = NearestNeighbors(metric='cosine', algorithm='brute')
    item_knn.fit(self.X.T)
    self.item_knn = item_knn
     

  def recommend(self, user_id, N=10):

    user_idx = self.uid2index[user_id]
    X = self.X

    # Find items this user has interacted with
    user_items = X[user_idx].nonzero()[1]
    
    # Initialize recommendation scores
    recommendations = np.zeros(X.shape[1])
    
    for item in user_items:
        # Find similar items
        distances, indices = self.item_knn.kneighbors(
            X.T[item].reshape(1, -1), 
            n_neighbors=11  # Including itself
        )
        
        # Convert distances to similarities and remove the item itself
        similarities = 1 - distances.flatten()
        similar_items = indices.flatten()
        
        # Skip the first one as it's the item itself
        for i, similar_item in enumerate(similar_items[1:]):
            # Weight by similarity
            recommendations[similar_item] += similarities[i+1]
    
    # Filter out items the user has already interacted with
    recommendations[user_items] = 0
    
    # Get top recommendations
    top_recommendations = recommendations.argsort()[-N:][::-1]

    top_recommendations = [self.index2nid[i] for i in top_recommendations]
    
    return top_recommendations