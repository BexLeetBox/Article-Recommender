from collections import Counter
import pandas as pd

class MostPopularRecommender:

    def __init__(self, behaviors_dataframe):
        """
        Initialize the recommender and compute article popularity.
        """
        self.behaviors = behaviors_dataframe
        self.most_popular_articles = []

    def train(self, N=10):
        """
        Train the model by counting how many times each article has been read,
        considering both user history and actual clicks from impressions.
        """
        all_read_articles = []

        for _, row in self.behaviors.iterrows():
            # Read articles from history
            if pd.notna(row["history"]):
                all_read_articles.extend(row["history"].split())

            # Read articles from impressions (only those clicked)
            if pd.notna(row["impressions"]):
                clicked_articles = [item.split("-")[0] for item in row["impressions"].split() if item.endswith("-1")]
                all_read_articles.extend(clicked_articles)

        # Count occurrences of each article
        article_read_counts = Counter(all_read_articles)

        # Get the most-read articles
        self.most_popular_articles = [article for article, _ in article_read_counts.most_common(N)]

    def recommend(self, user_id=None, N=10):
        """
        Recommend the top-N most popular articles.
        """
        return self.most_popular_articles[:N] if self.most_popular_articles else []