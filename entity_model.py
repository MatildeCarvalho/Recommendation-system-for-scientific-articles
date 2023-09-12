import pandas as pd
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity

class ArticleRecommendationSystem:
    def __init__(self, train_file, test_file, articles_file):
        """
        Initialize the recommendation system with data from CSV files.

        Args:
            train_file (str): Path to the training data CSV file.
            test_file (str): Path to the test data CSV file.
            articles_file (str): Path to the articles data CSV file.
        """
        self.train = self.load_dataframe(train_file)
        self.test = self.load_dataframe(test_file)
        self.articles_df = self.load_dataframe(articles_file)
        self.user_profiles = {}

    def load_dataframe(self, file_path):
        """
        Load a CSV file into a pandas DataFrame.

        Args:
            file_path (str): Path to the CSV file.

        Returns:
            pd.DataFrame: Loaded DataFrame.
        """
        return pd.read_csv(file_path)

    def get_user_ratings(self, user_id, train):
        """
        Get the articles and ratings for a user from the training or test data.

        Args:
            user_id (int): User ID.
            train (pd.DataFrame): Training data DataFrame our test.

        Returns:
            list, list: List of article IDs, list of user ratings.
        """
        user_data = train[train['user'] == user_id]
        articles_ids = user_data['article'].tolist()
        user_ratings = user_data['rating'].tolist()
        return articles_ids, user_ratings

    def get_entities_for_articles(self, article_ids):
        """
        Get entities associated with a list of article IDs.

        Args:
            article_ids (list): List of article IDs.

        Returns:
            list: List of entity lists for each article.
        """
        entities_lists = []

        for article_id in article_ids:
            article_data = self.articles_df[self.articles_df['doc.id'] == article_id]
            entities_sequence = article_data['entities'].values[0]
            entities_list = re.findall(r"'([^']+)'", entities_sequence)
            entities_lists.append(entities_list)

        return entities_lists

    def get_all_user_entities(self, user_id):
        """
        Get all unique entities associated with a specific user.

        Args:
            user_id (int): User ID.

        Returns:
            list: List of unique entities associated with the user.
        """
        user_data = self.train[self.train['user'] == user_id]
        entities_for_articles = user_data['entities'].tolist()
        all_entities = []

        for sequence in entities_for_articles:
            if isinstance(sequence, str):
                entities = re.findall(r"'([^']+)'", sequence)
                all_entities.extend(entities)

        return all_entities

    def criar_matriz_articles_entidades(self, articles, entidades_unicas, sequencias_entidades):
        """
        Create a matrix of articles and their associated entities (articles*entities).

        Args:
            articles (list): List of article IDs.
            entidades_unicas (list): List of unique entities.
            sequencias_entidades (list): List of entity sequences for articles.

        Returns:
            np.ndarray: A matrix representing the presence of entities in articles.
        """
        matriz_articles_entidades = np.zeros((len(articles), len(entidades_unicas)), dtype=int)

        for i, article in enumerate(articles):
            for j, entidade in enumerate(entidades_unicas):
                matriz_articles_entidades[i][j] = sequencias_entidades[i].count(entidade)

        return matriz_articles_entidades

    def weighted_entity_matrix(self, articles_matrix, user_ratings):
        """
        Calculate a weighted entity matrix based on article matrix and user ratings.

        Args:
            articles_matrix (np.ndarray): Matrix representing articles and entities.
            user_ratings (list): List of user ratings for articles.

        Returns:
            np.ndarray: Weighted entity matrix.
        """
        articles_matrix = np.array(articles_matrix)
        user_ratings = np.array(user_ratings)
        weighted_matrix = articles_matrix * user_ratings[:, np.newaxis]
        return weighted_matrix

    def user_profile(self, weighted_matrix):
        """
        Calculate the user profile based on the weighted entity matrix.

        Args:
            weighted_matrix (np.ndarray): Weighted entity matrix.

        Returns:
            np.ndarray: User profile.
        """
        user_profile = np.sum(weighted_matrix, axis=0)
        norm = np.linalg.norm(user_profile)

        if norm == 0:
            return user_profile

        normalized_user_profile = user_profile / norm
        return normalized_user_profile

    def weighted_articles_matrix(self, test_articles_matrix, user_profile):
        """
        Calculate a weighted articles matrix for recommendation.

        Args:
            test_articles_matrix (np.ndarray): Matrix for test articles.
            user_profile (np.ndarray): User profile.

        Returns:
            np.ndarray: Weighted articles matrix.
        """
        weighted_matrix = test_articles_matrix * user_profile
        return weighted_matrix

    def train_user(self, user_id):
        """
        Train the recommendation system for a specific user.

        Args:
            user_id (int): User ID.
        """
        articles_ids_list, user_ratings = self.get_user_ratings(user_id, self.train)
        entities_for_articles = self.get_entities_for_articles(articles_ids_list)
        articles_matrix = self.criar_matriz_articles_entidades(articles_ids_list, self.get_all_user_entities(user_id), entities_for_articles)
        weighted_matrix = self.weighted_entity_matrix(articles_matrix, user_ratings)
        user_profiles = self.user_profile(weighted_matrix)
        self.user_profiles[user_id] = user_profiles

    def recommend_articles(self, user_id, num_recommendations=5):
        """
        Recommend articles to a user based on their profile.

        Args:
            user_id (int): User ID.
            num_recommendations (int): Number of articles to recommend (default: 5).

        Returns:
            list: List of recommended article IDs.
        """
        if user_id not in self.user_profiles:
            return []

        articles_ids_list_TEST, user_ratings = self.get_user_ratings(user_id, self.test)
        entities_for_articles_TEST = self.get_entities_for_articles(articles_ids_list_TEST)
        candidate_articles_matrix = self.criar_matriz_articles_entidades(articles_ids_list_TEST, self.get_all_user_entities(user_id), entities_for_articles_TEST)
        test_articles_matrix = self.weighted_entity_matrix(candidate_articles_matrix, user_ratings)

        similarities = cosine_similarity([self.user_profiles[user_id]], test_articles_matrix)
        recommended_indices = similarities.argsort()[0][::-1]
        recommended_article_ids = [articles_ids_list_TEST[i] for i in recommended_indices[:num_recommendations]]
        return recommended_article_ids

# Exemplo de uso:
#recommendation_system = ArticleRecommendationSystem('Train_teste_split\\train_set.csv', 'Train_teste_split\\test_set.csv', 'file_final\\articles_combined.csv')
#user_id = 0
#recommendation_system.train_user(user_id)
#recommended_article_ids = recommendation_system.recommend_articles(user_id, num_recommendations=5)
#print("Artigos recomendados:", recommended_article_ids)
