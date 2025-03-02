from collections import Counter

class MostPopularGeneralRecommender:

    def __init__(self, behaviors_dataframe):
        """
        Initialize the recommender with dataset path.
        """
        self.behaviors = behaviors_dataframe
        self.most_popular_articles = []

    def train(self, N=10):
        """
        Train the baseline model using user history.
        """
        df = self.behaviors.dropna(subset=["history"]) # Users with no history can be removed
        all_clicked_articles = df["history"].str.split().explode()
        article_popularity = Counter(all_clicked_articles)
        self.most_popular_articles = [article for article, _ in article_popularity.most_common(N)]

    def recommend(self, user_id=None, N=10):
        """
        Recommend the top-N most popular articles.
        """
        return self.most_popular_articles[:N] if self.most_popular_articles else []