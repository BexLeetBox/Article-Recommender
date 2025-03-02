from collections import Counter

class MostPopularPersonalizedRecommender:

    def __init__(self, behaviors_dataframe):
        """
        Initialize the recommender with dataset path.
        """
        self.behaviors = behaviors_dataframe
        self.most_popular_articles = []
        self.user_histories = self.behaviors.set_index("user_id")["history"].to_dict()

    def train(self, N=10):
        """
        Train the baseline model using user history.
        """
        all_clicked_articles = self.behaviors["history"].str.split().explode()
        article_popularity = Counter(all_clicked_articles)
        self.most_popular_articles = [article for article, _ in article_popularity.most_common(N)]

    def recommend(self, user_id, N=10):
        user_history = str(self.user_histories.get(user_id, ""))
        user_read_articles = set(user_history.split()) if user_history else set()
        unseen_articles = [article for article in self.most_popular_articles if article not in user_read_articles]

        return unseen_articles[:N]