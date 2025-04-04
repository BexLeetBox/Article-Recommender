import sys

import numpy as np
import pandas as pd
from IPython.display import clear_output

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
    """
    ndcg_scores = []
    match_stack = []

    for _, row in behaviors_df.iterrows():
        user_id = row["user_id"]
        if pd.isna(row["impressions"]):
            continue

        # Extract clicked articles
        actual_clicked = {item.split("-")[0] for item in row["impressions"].split() if item.endswith("-1")}
        if not actual_clicked:
            continue

        # Get recommendations
        recommended = recommender.recommend(user_id, N=K) or []
        recommended = recommended[:K]

        matches = set(recommended) & set(actual_clicked)
        if matches:
            match_stack.append((user_id, matches))  # Store user & matched items



        clear_output(wait=True)

        # Debug: Print recommendations vs. actual clicks
        print(f"\n User {user_id}")
        print(f"    Clicked: {actual_clicked}")
        print(f"    Recommended: {recommended}")
        #print(f"    Matches: {set(recommended) & set(actual_clicked)}")
        print(f"   🎯 Matches: {matches}")
        if not set(recommended) & set(actual_clicked):
            print("❌ No matches found for this user!")

        sys.stdout.flush()

        # Compute scores
        precision_scores.append(precision_at_k(recommended, actual_clicked, K))
        recall_scores.append(recall_at_k(recommended, actual_clicked, K))
        ndcg_scores.append(ndcg_at_k(recommended, actual_clicked, K))
        auc_scores.append(auc_at_k(recommended, actual_clicked, K))
        mrr_scores.append(mrr_at_k(recommended, actual_clicked, K))

    avg_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0.0
    avg_auc = np.mean(auc_scores) if auc_scores else 0.0
    avg_mrr = np.mean(mrr_scores) if mrr_scores else 0.0

    return avg_precision, avg_recall, avg_ndcg
