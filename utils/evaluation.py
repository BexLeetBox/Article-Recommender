import numpy as np
import pandas as pd

def precision_at_k(recommended, actual_clicked, K=10):
    """
    Computes Precision@K (how many top-K recommendations were actually clicked).
    """
    if not actual_clicked or K == 0:
        return 0.0
    return (len(set(recommended[:K]) & set(actual_clicked))) / K

def recall_at_k(recommended, actual_clicked, K=10):
    """
    Computes Recall@K (how many of the relevant items were recommended).
    """
    if not actual_clicked:
        return 0.0
    return (len(set(recommended[:K]) & set(actual_clicked))) / len(actual_clicked)

def dcg_at_k(recommended, actual_clicked, K=10):
    """
    Computes Discounted Cumulative Gain (DCG) at K.
    """
    return sum(1 / np.log2(i + 2) for i, item in enumerate(recommended[:K]) if item in actual_clicked)

def idcg_at_k(actual_clicked, K=10):
    """
    Computes Ideal Discounted Cumulative Gain (IDCG) at K.
    """
    num_relevant = min(len(actual_clicked), K)
    return sum(1 / np.log2(i + 2) for i in range(num_relevant)) or 1  # Avoid division by zero

def ndcg_at_k(recommended, actual_clicked, K=10):
    """
    Computes Normalized Discounted Cumulative Gain (NDCG) at K.
    """
    return dcg_at_k(recommended, actual_clicked, K) / idcg_at_k(actual_clicked, K)

def evaluate_model(recommender, behaviors_df, K=10):
    """
    Evaluates a recommender model on a dataset.
    Uses the `impressions` column to determine relevant items.
    Returns average Precision@K and Recall@K.
    """
    precision_scores = []
    recall_scores = []
    ndcg_scores = []

    for _, row in behaviors_df.iterrows():
        user_id = row["user_id"]
        if pd.isna(row["impressions"]):  # Skip users with no impressions
            continue

        # Extract clicked articles from impressions (those marked with `-1`)
        actual_clicked = {item.split("-")[0] for item in row["impressions"].split() if item.endswith("-1")}

        if not actual_clicked:
            continue

        # Get recommendations from the model
        recommended = recommender.recommend(user_id, N=K) or []
        recommended = recommended[:K]

        # Compute precision and recall
        precision_scores.append(precision_at_k(recommended, actual_clicked, K))
        recall_scores.append(recall_at_k(recommended, actual_clicked, K))
        ndcg_scores.append(ndcg_at_k(recommended, actual_clicked, K))

    avg_precision = np.mean(precision_scores) if precision_scores else 0.0
    avg_recall = np.mean(recall_scores) if recall_scores else 0.0
    avg_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0.0

    return avg_precision, avg_recall, avg_ndcg