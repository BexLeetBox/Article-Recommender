import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

class ALSRecommender:
    def __init__(self, behaviors_df, num_factors=20, alpha=40, reg=0.1, num_iters=10):
        """
        Initialize the ALS recommender using the behaviors DataFrame.
        Builds the user-item interaction matrix internally and initializes latent factors.

        behaviors_df: DataFrame containing training data (with at least "user_id" and "history" columns).
        num_factors: Number of latent factors.
        alpha: Confidence scaling factor for implicit feedback.
        reg: Regularization parameter.
        num_iters: Number of ALS iterations.
        """
        self.behaviors_df = behaviors_df
        self.num_factors = num_factors
        self.alpha = alpha
        self.reg = reg
        self.num_iters = num_iters

        # Build the interaction matrix internally
        self._build_interaction_matrix()
        self.num_users, self.num_items = self.interaction_matrix.shape

        # Initialize user and item latent factor matrices with small random values.
        self.U = np.random.normal(scale=0.01, size=(self.num_users, self.num_factors))
        self.V = np.random.normal(scale=0.01, size=(self.num_items, self.num_factors))

    def _build_interaction_matrix(self):
        """
        Processes the behaviors DataFrame to create a sparse user-item interaction matrix.
        The "history" column is used to determine which articles a user has interacted with.
        """
        user_articles = {}
        for _, row in self.behaviors_df.iterrows():
            user = row["user_id"]
            history = row["history"]
            if pd.isna(history):
                continue
            articles = history.split()
            if user in user_articles:
                user_articles[user].update(articles)
            else:
                user_articles[user] = set(articles)

        # Map users and articles to indices.
        self.user_ids = list(user_articles.keys())
        self.user_to_idx = {user: idx for idx, user in enumerate(self.user_ids)}

        all_articles = set()
        for articles in user_articles.values():
            all_articles.update(articles)
        self.all_articles = list(all_articles)
        self.article_to_idx = {article: idx for idx, article in enumerate(self.all_articles)}

        # Build the sparse interaction matrix (binary: 1 if user interacted with article).
        data = []
        row_indices = []
        col_indices = []
        for user, articles in user_articles.items():
            u_idx = self.user_to_idx[user]
            for article in articles:
                row_indices.append(u_idx)
                col_indices.append(self.article_to_idx[article])
                data.append(1)

        self.interaction_matrix = csr_matrix(
            (data, (row_indices, col_indices)),
            shape=(len(self.user_ids), len(self.all_articles))
        )

    def train(self):
        """
        Trains the ALS model using alternating least squares updates.
        """
        num_factors = self.num_factors
        reg_eye = self.reg * np.eye(num_factors)

        for iteration in range(self.num_iters):
            # --- Update user factors ---
            for u in range(self.num_users):
                # Get indices of items user u has interacted with.
                start_ptr, end_ptr = self.interaction_matrix.indptr[u], self.interaction_matrix.indptr[u + 1]
                item_indices = self.interaction_matrix.indices[start_ptr:end_ptr]
                if len(item_indices) == 0:
                    continue

                A = self.V.T.dot(self.V) + reg_eye
                b = np.zeros(num_factors)
                for i in item_indices:
                    v_i = self.V[i]
                    A += self.alpha * np.outer(v_i, v_i)
                    b += (1 + self.alpha) * v_i
                self.U[u] = np.linalg.solve(A, b)

            # --- Update item factors ---
            # For efficiency, convert to CSC for column slicing.
            interaction_csc = self.interaction_matrix.tocsc()
            for i in range(self.num_items):
                start_ptr, end_ptr = interaction_csc.indptr[i], interaction_csc.indptr[i + 1]
                user_indices = interaction_csc.indices[start_ptr:end_ptr]
                if len(user_indices) == 0:
                    continue

                A = self.U.T.dot(self.U) + reg_eye
                b = np.zeros(num_factors)
                for u in user_indices:
                    u_factor = self.U[u]
                    A += self.alpha * np.outer(u_factor, u_factor)
                    b += (1 + self.alpha) * u_factor
                self.V[i] = np.linalg.solve(A, b)

            print(f"Iteration {iteration + 1}/{self.num_iters} completed.")

    def recommend(self, user_id, N=10):
        """
        Generate N recommendations for a given user.
        user_id: The actual user ID as in the training data.
        Returns a list of recommended article IDs (not indices).
        """
        if user_id not in self.user_to_idx:
            return []

        u_idx = self.user_to_idx[user_id]
        user_vector = self.U[u_idx]
        scores = self.V.dot(user_vector)

        # Get items the user already interacted with.
        start_ptr, end_ptr = self.interaction_matrix.indptr[u_idx], self.interaction_matrix.indptr[u_idx + 1]
        interacted_indices = set(self.interaction_matrix.indices[start_ptr:end_ptr])

        # Rank items by score.
        ranked_item_indices = np.argsort(-scores)
        recommendations = [self.all_articles[i] for i in ranked_item_indices if i not in interacted_indices]
        return recommendations[:N]

        # --- Evaluation Functions ---
        def dcg_at_k(self, recommended, actual_clicked, K=5):
            """
            Computes Discounted Cumulative Gain (DCG) at K.
            """
            return sum(1 / np.log2(i + 2) for i, item in enumerate(recommended[:K]) if item in actual_clicked)

        def idcg_at_k(self, actual_clicked, K=5):
            """
            Computes Ideal Discounted Cumulative Gain (IDCG) at K.
            """
            num_relevant = min(len(actual_clicked), K)
            return sum(1 / np.log2(i + 2) for i in range(num_relevant)) or 1

        def ndcg_at_k(self, recommended, actual_clicked, K=5):
            """
            Computes Normalized Discounted Cumulative Gain (NDCG) at K.
            """
            return self.dcg_at_k(recommended, actual_clicked, K) / self.idcg_at_k(actual_clicked, K)

        def evaluate(self, validation_behaviors_df, K=5):
            """
            Evaluates the ALS recommender on a validation behaviors DataFrame.
            For each row, extracts actual clicked articles from the 'impressions' column,
            computes NDCG at K based on recommendations, and returns the average NDCG score.
            """
            ndcg_scores = []
            for _, row in validation_behaviors_df.iterrows():
                user_id = row["user_id"]
                # Skip rows without impressions.
                if pd.isna(row["impressions"]):
                    continue

                # Extract actual clicked articles from impressions.
                actual_clicked = {
                    item.split("-")[0] for item in row["impressions"].split()
                    if item.split("-")[1] == "1"
                }
                if not actual_clicked:
                    continue

                # Only evaluate users that exist in the training data.
                if user_id not in self.user_to_idx:
                    continue

                recommended = self.recommend(user_id, N=K)
                ndcg = self.ndcg_at_k(recommended, actual_clicked, K)
                ndcg_scores.append(ndcg)

            return np.mean(ndcg_scores) if ndcg_scores else 0.0


