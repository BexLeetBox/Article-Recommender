import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# ----- Standalone helper functions -----

def process_interactions(behavior_data, decay_rate=0.1, use_timestamps=True):
    """
    Processes behavior data to extract user-item interactions with optional time weighting.
    """
    print("Processing interactions...")
    interactions = []
    has_timestamp = 'timestamp' in behavior_data.columns and use_timestamps

    if has_timestamp:
        if not pd.api.types.is_datetime64_any_dtype(behavior_data['timestamp']):
            behavior_data['timestamp'] = pd.to_datetime(behavior_data['timestamp'], errors='coerce')
        reference_time = behavior_data['timestamp'].max()

    for _, row in tqdm(behavior_data.iterrows(), total=len(behavior_data), desc="Extracting interactions"):
        user_id = row['user_id']
        if pd.isna(row['history']):
            continue
        history = row['history'].split()
        time_weight = 1.0
        if has_timestamp and pd.notna(row.get('timestamp')):
            days_diff = (reference_time - row['timestamp']).total_seconds() / (24 * 3600)
            time_weight = np.exp(-decay_rate * days_diff)
        for position, article_id in enumerate(history):
            position_weight = 1.0 - 0.8 * (len(history) - 1 - position) / max(1, len(history) - 1)
            weight = position_weight * time_weight
            interactions.append({
                'user_id': user_id,
                'article_id': article_id,
                'weight': weight
            })

    interactions_df = pd.DataFrame(interactions)
    if not interactions_df.empty:
        interactions_df = interactions_df.groupby(['user_id', 'article_id'])['weight'].sum().reset_index()
    return interactions_df

def create_matrices(interactions_df):
    """
    Creates the sparse user-item matrix and corresponding mappings.
    """
    print("Creating matrices...")
    user_ids = interactions_df['user_id'].unique()
    article_ids = interactions_df['article_id'].unique()
    user_id_to_idx = {uid: i for i, uid in enumerate(user_ids)}
    article_id_to_idx = {aid: i for i, aid in enumerate(article_ids)}
    user_indices = interactions_df['user_id'].map(user_id_to_idx).values
    article_indices = interactions_df['article_id'].map(article_id_to_idx).values
    interaction_values = interactions_df['weight'].values
    user_item_matrix = csr_matrix(
        (interaction_values, (user_indices, article_indices)),
        shape=(len(user_ids), len(article_ids))
    )
    item_popularity_array = np.array(user_item_matrix.sum(axis=0)).flatten()
    item_popularity_array = item_popularity_array / np.sum(item_popularity_array)
    idx_to_user_id = {i: uid for uid, i in user_id_to_idx.items()}
    idx_to_article_id = {i: aid for aid, i in article_id_to_idx.items()}
    return user_item_matrix, user_id_to_idx, article_id_to_idx, idx_to_user_id, idx_to_article_id, item_popularity_array

def create_item_similarity_matrix(user_item_matrix, batch_size=1000):
    """
    Computes the item-item similarity matrix using normalized cosine similarity.
    """
    print("Computing item similarity matrix...")
    item_user_matrix = user_item_matrix.T.tocsr()
    normalized_item_matrix = normalize(item_user_matrix, norm='l2', axis=1)
    n_items = normalized_item_matrix.shape[0]
    item_similarities = csr_matrix((n_items, n_items))

    for start_idx in range(0, n_items, batch_size):
        end_idx = min(start_idx + batch_size, n_items)
        current_batch = normalized_item_matrix[start_idx:end_idx]
        batch_similarity = cosine_similarity(
            current_batch,
            normalized_item_matrix,
            dense_output=False
        ).tolil()
        # Zero out self-similarity
        for i in range(end_idx - start_idx):
            batch_similarity[i, start_idx + i] = 0
        batch_similarity = batch_similarity.tocoo()
        item_similarities = item_similarities + csr_matrix(
            (batch_similarity.data,
             (batch_similarity.row + start_idx, batch_similarity.col)),
            shape=(n_items, n_items)
        )
    return item_similarities

def generate_recommendations(user_id, train_data, user_item_matrix, item_similarities,
                              user_id_to_idx, article_id_to_idx, item_popularity,
                              N=5, candidate_articles=None):
    """
    Generate top-N recommendations with custom aggregation and cold-start handling.
    """
    scores = {}
    user_history = []

    if train_data is not None:
        user_rows = train_data[train_data['user_id'] == user_id]
        if not user_rows.empty:
            history_str = user_rows.iloc[0]['history']
            if pd.notna(history_str) and history_str.strip() != "":
                user_history = history_str.split()

    if candidate_articles is None:
        candidate_articles = list(article_id_to_idx.keys())

    if user_id in user_id_to_idx:
        user_idx = user_id_to_idx[user_id]
        for article_id in candidate_articles:
            if article_id not in article_id_to_idx:
                continue

            article_idx = article_id_to_idx[article_id]

            if user_item_matrix[user_idx, article_idx] > 0:
                score = 0.0
            else:
                if user_history:
                    sim_scores = []
                    for hist_id in user_history:
                        if hist_id in article_id_to_idx:
                            hist_idx = article_id_to_idx[hist_id]
                            sim = item_similarities[hist_idx, article_idx]
                            if sim > 0:
                                sim_scores.append(sim)

                    if sim_scores:
                        # Dynamic weighting: more history -> more CF trust
                        pop_score = item_popularity[article_idx]
                        weight_cf = min(0.6 + 0.05 * len(sim_scores), 0.9)
                        weight_pop = 1 - weight_cf
                        score = weight_cf * max(sim_scores) + weight_pop * pop_score
                    else:
                        # No good similarities, fallback to popularity
                        score = item_popularity[article_idx]
                else:
                    # No history at all
                    score = item_popularity[article_idx]

                # Apply tanh transformation
                score = np.tanh(3 * (score - 0.5)) * 0.5 + 0.5

            scores[article_id] = score

    else:
        # Cold-start user
        for article_id in candidate_articles:
            if article_id not in article_id_to_idx:
                continue

            article_idx = article_id_to_idx[article_id]
            pop_score = item_popularity[article_idx]
            score = 0.5 * pop_score + 0.5 * np.random.uniform(0, 0.1)

            # Apply tanh
            score = np.tanh(3 * (score - 0.5)) * 0.5 + 0.5
            scores[article_id] = score

    recommended = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:N]
    return [article_id for article_id, _ in recommended]



# ----- The Hybrid Recommender class using composition -----

class HybridRecommender:
    def __init__(self, batch_size=1000, decay_rate=0.1, sigmoid_multiplier=5):
        self.batch_size = batch_size
        self.decay_rate = decay_rate
        self.sigmoid_multiplier = sigmoid_multiplier
        self.train_data = None
        self.interactions = None
        self.user_item_matrix = None
        self.user_id_to_idx = {}
        self.article_id_to_idx = {}
        self.item_popularity = None
        self.item_similarities = None

    def train(self, train_file, use_timestamps=True):
        columns = ["impression_id", "user_id", "timestamp", "history", "impressions"]
        self.train_data = pd.read_csv(train_file, sep='\t', names=columns)
        self.interactions = process_interactions(self.train_data, decay_rate=self.decay_rate, use_timestamps=use_timestamps)
        (self.user_item_matrix,
         self.user_id_to_idx,
         self.article_id_to_idx,
         _,
         _,
         self.item_popularity) = create_matrices(self.interactions)
        self.item_similarities = create_item_similarity_matrix(self.user_item_matrix, batch_size=self.batch_size)
        print("Training completed.")

    def recommend(self, user_id, N=5, candidate_articles=None):
        return generate_recommendations(user_id, self.train_data, self.user_item_matrix,
                                        self.item_similarities, self.user_id_to_idx,
                                        self.article_id_to_idx, self.item_popularity, N=N,
                                        candidate_articles=candidate_articles)