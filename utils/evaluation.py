import numpy as np
import pandas as pd

def dcg_at_k(recommended, actual_clicked, K=5):
    """
    Computes Discounted Cumulative Gain (DCG) at K.
    """
    return sum(1 / np.log2(i + 2) for i, item in enumerate(recommended[:K]) if item in actual_clicked)

def idcg_at_k(actual_clicked, K=5):
    """
    Computes Ideal Discounted Cumulative Gain (IDCG) at K.
    """
    num_relevant = min(len(actual_clicked), K)
    return sum(1 / np.log2(i + 2) for i in range(num_relevant)) or 1

def ndcg_at_k(recommended, actual_clicked, K=5):
    """
    Computes Normalized Discounted Cumulative Gain (NDCG) at K.
    """
    return dcg_at_k(recommended, actual_clicked, K) / idcg_at_k(actual_clicked, K)


def auc_at_k(recommended, actual_clicked, K=5):
    """
    Computes AUC at K.

    Here we treat the top-K recommended list as a binary ranking vector (1 if clicked, else 0)
    and compute the fraction of positive-negative pairs that are correctly ordered.
    """
    recommended = recommended[:K]
    relevance = [1 if item in actual_clicked else 0 for item in recommended]
    num_positives = sum(relevance)
    num_negatives = len(relevance) - num_positives

    if num_positives == 0:
        return 0.0
    if num_negatives == 0:
        return 1.0

    correct_pairs = 0
    total_pairs = num_positives * num_negatives
    for i in range(len(relevance)):
        if relevance[i] == 1:
            for j in range(len(relevance)):
                if relevance[j] == 0 and i < j:
                    correct_pairs += 1
    return correct_pairs / total_pairs

def mrr_at_k(recommended, actual_clicked, K=5):
    """
    Computes Mean Reciprocal Rank (MRR) at K.
    """
    recommended = recommended[:K]
    for i, item in enumerate(recommended):
        if item in actual_clicked:
            return 1.0 / (i + 1)
    return 0.0

def evaluate_model(recommender, behaviors_df, K=10):
    """
    Evaluates a recommender model on a dataset.
    Uses the `impressions` column to determine relevant items.
    """
    ndcg_scores = []
    auc_scores = []
    mrr_scores = []

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

        ndcg_scores.append(ndcg_at_k(recommended, actual_clicked, K))
        auc_scores.append(auc_at_k(recommended, actual_clicked, K))
        mrr_scores.append(mrr_at_k(recommended, actual_clicked, K))

    avg_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0.0
    avg_auc = np.mean(auc_scores) if auc_scores else 0.0
    avg_mrr = np.mean(mrr_scores) if mrr_scores else 0.0

    return avg_ndcg, avg_auc, avg_mrr