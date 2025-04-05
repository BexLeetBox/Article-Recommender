import numpy as np
import sys
import os

from sklearn.decomposition import NMF

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".." ))
)

from collaborative.collab_utils import create_x, preprocess_clicked, read_impressions_tsv

class MFRecommender:
  
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
    # Number of latent factors
    n_factors = 50

    # Initialize NMF model
    nmf_model = NMF(n_components=n_factors, init='random', random_state=42)

    # Fit the model to the user-item matrix
    self.user_factors = nmf_model.fit_transform(self.X)
    self.item_factors = nmf_model.components_
        

  def recommend(self, user_id, N=10):

    if not self.user_exist(user_id):
      print(f"user {user_id} does not exist")
      return []
    user_index = self.uid2index[user_id]


    # Predict ratings for all items
    user_vector = self.user_factors[user_index].reshape(1, -1)
    predicted_ratings = np.dot(user_vector, self.item_factors)
    
    # Set already interacted items to zero
    user_items = self.X[user_index].nonzero()[1]
    predicted_ratings[0, user_items] = 0
    
    # Get top recommendations
    top_recommendations = predicted_ratings[0].argsort()[-N:][::-1]
    top_recommendations = [self.index2nid[i] for i in top_recommendations]
    
    
    return top_recommendations