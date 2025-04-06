from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import sys
import os


sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".." ))
)

from collaborative.collab_utils import create_x, preprocess_history, read_impressions_tsv

class UserSimilarityRecommender:
  
  def user_exist(self, user_id):
     return user_id in self.uid2index

  def __init__(self, df=None, preprocessed=False):
    if df is None:
      df = read_impressions_tsv("../MIND/train/behaviors.tsv")

    if not preprocessed:
      df = df.dropna()
      df = df.copy()
      df["news_id"] = df["history"].apply(lambda x: x.split())
      df = df.explode("news_id").reset_index()
      df["click"] = 1
      df = df[["user_id", "news_id", "click"]]

    self.df = df


  def train(self):
    self.X, self.uid2index, _, _, self.index2nid = create_x(self.df)
    self.user_similarity = self.calculate_user_similarity()
     

  def recommend(self, user_id=None, N=10):
    if user_id not in self.uid2index: 
       print(f"Cannot make recommendations for {user_id}, not found in matrix")
       return []
    
    user_idx = self.uid2index[user_id]

    neighborhood = self.create_neighborhood(user_idx)

    scores = self.calculate_recommendation_scores(user_idx, neighborhood)

    recommendations = self.get_top_recommendations(scores, N, self.index2nid)

    return recommendations

  def calculate_user_similarity(self):
      # Calculate user similarity for pairs using cosine similarity
      user_similarity = cosine_similarity(self.X)
      
      # Fill diagonal with 0 to ensure that pairs (i, i) are not counted
      np.fill_diagonal(user_similarity, 0)
      
      return user_similarity

  def create_neighborhood(self, user_idx, similarity_threshold=0.1, neighborhood_size=10):
      user_similarities = self.user_similarity[user_idx]
      
      similar_users = np.where(user_similarities > similarity_threshold)[0]
      sorted_similar_users = sorted(similar_users, key=lambda x: user_similarities[x], reverse=True)
      
      return sorted_similar_users[:neighborhood_size]


  def calculate_recommendation_scores(self, user_idx, neighborhood, weighted=True):

      interaction_matrix = self.X
      n_news = interaction_matrix.shape[1]
      scores = np.zeros(n_news)
      
      user_interactions = interaction_matrix[user_idx].toarray().flatten()
      already_interacted = set(np.where(user_interactions > 0)[0])
      
      for neighbor_idx in neighborhood:
          neighbor_interactions = interaction_matrix[neighbor_idx].toarray().flatten()
          
          if weighted:
              similarity = self.user_similarity[user_idx, neighbor_idx]
              for news_idx in np.where(neighbor_interactions > 0)[0]:
                  if news_idx not in already_interacted:
                      scores[news_idx] += similarity * neighbor_interactions[news_idx]
          else:
              # Unweighted - just count interactions
              for news_idx in np.where(neighbor_interactions > 0)[0]:
                  if news_idx not in already_interacted:
                      scores[news_idx] += 1
      
      return scores


  def get_top_recommendations(self, scores, n_recommendations, index2nid):
      # Sort scores and get indices of top N recommendations
      top_indices = np.argsort(scores)[::-1][:n_recommendations]
      
      # Convert indices to news IDs
      recommendations = [index2nid[idx] for idx in top_indices if scores[idx] > 0]
      
      return recommendations