#FUNCIONA

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ArticleRecommendationSystemTFIDF:
    def __init__(self, train_file, test_file, articles_file):
        """
        Inicializa uma instância do sistema de recomendação de artigos baseado em TF-IDF.

        Args:
            train_file (str): O caminho para o arquivo CSV contendo o conjunto de treinamento.
            test_file (str): O caminho para o arquivo CSV contendo o conjunto de teste.
            articles_file (str): O caminho para o arquivo CSV contendo os dados dos artigos.
        """
        self.train = self.load_dataframe(train_file)
        self.test = self.load_dataframe(test_file)
        self.articles_df = self.load_dataframe(articles_file)
        self.user_profiles = {}
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')  # Use a lista de stop words em inglês

    def load_dataframe(self, file_path):
        """
        Carrega um arquivo CSV em um DataFrame do Pandas.

        Args:
            file_path (str): O caminho para o arquivo CSV a ser carregado.

        Returns:
            pd.DataFrame: O DataFrame contendo os dados do arquivo CSV.
        """
        return pd.read_csv(file_path)

    def train_user(self, user_id):
        """
        Treina o modelo para um usuário específico com base em seus interações no conjunto de treinamento.

        Args:
            user_id (int): O ID do usuário para o qual o modelo deve ser treinado.
        """
        # Obtenha os resumos (abstracts) dos artigos que o usuário interagiu no conjunto de treinamento
        articles_ids_list, user_ratings = self.get_user_ratings(user_id, self.train)
        article_abstracts = self.get_abstracts_for_articles(articles_ids_list)

        # Ajuste o modelo TF-IDF aos resumos dos artigos
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(article_abstracts)

        # Calcule o perfil do usuário com base no TF-IDF
        user_profile = np.array(tfidf_matrix.sum(axis=0))[0]

        # Normalize o perfil do usuário
        norm = np.linalg.norm(user_profile)
        if norm != 0:
            user_profile /= norm

        self.user_profiles[user_id] = user_profile

    def recommend_articles(self, user_id, num_recommendations=5):
        """
        Recomenda artigos para um usuário com base no modelo treinado.

        Args:
            user_id (int): O ID do usuário para o qual a recomendação deve ser feita.
            num_recommendations (int): O número de artigos a serem recomendados (padrão: 5).

        Returns:
            list: Uma lista de IDs dos artigos recomendados.
        """
        if user_id not in self.user_profiles:
            return []

        # Obtenha os resumos (abstracts) dos artigos no conjunto de teste
        articles_ids_list_TEST, user_ratings = self.get_user_ratings(user_id, self.test)
        article_abstracts_TEST = self.get_abstracts_for_articles(articles_ids_list_TEST)

        # Transforme os resumos dos artigos do conjunto de teste em vetores TF-IDF
        test_articles_matrix = self.tfidf_vectorizer.transform(article_abstracts_TEST)

        # Calcule a similaridade de cosseno entre o perfil do usuário e os artigos de teste
        similarities = cosine_similarity(self.user_profiles[user_id].reshape(1, -1), test_articles_matrix)

        # Obtenha os índices dos artigos recomendados com base na similaridade do cosseno
        recommended_indices = similarities.argsort()[0][::-1]

        # Se houver menos artigos do que 'num_recommendations', ajuste o valor de 'num_recommendations'
        num_recommendations = min(num_recommendations, len(articles_ids_list_TEST))

        # Selecione os IDs dos artigos recomendados
        recommended_article_ids = [articles_ids_list_TEST[i] for i in recommended_indices[:num_recommendations]]
        
        return recommended_article_ids

    def get_user_ratings(self, user_id, train):
        """
        Obtém os IDs dos artigos e as avaliações que um usuário deu no conjunto de treinamento.

        Args:
            user_id (int): O ID do usuário.
            train (pd.DataFrame): O DataFrame contendo o conjunto de treinamento.

        Returns:
            tuple: Uma tupla contendo dois elementos:
                - Uma lista de IDs dos artigos que o usuário avaliou.
                - Uma lista de avaliações que o usuário deu aos artigos correspondentes.
        """
        user_data = train[train['user'] == user_id]
        articles_ids = user_data['article'].tolist()
        user_ratings = user_data['rating'].tolist()
        return articles_ids, user_ratings

    def get_abstracts_for_articles(self, article_ids):
        """
        Obtém os resumos (abstracts) dos artigos com base em seus IDs.

        Args:
            article_ids (list): Uma lista de IDs dos artigos.

        Returns:
            list: Uma lista de resumos (abstracts) dos artigos correspondentes.
        """
        abstracts = []

        for article_id in article_ids:
            article_data = self.articles_df[self.articles_df['doc.id'] == article_id]
            abstract = article_data['clean_abstract'].values[0]
            abstracts.append(abstract)

        return abstracts


# Exemplo de uso:
#recommendation_system = ArticleRecommendationSystemTFIDF('Train_teste_split\\train_set.csv', 'Train_teste_split\\test_set.csv', 'file_final\\articles_combined.csv')
#user_id = 0
#recommendation_system.train_user(user_id)
#recommended_article_ids = recommendation_system.recommend_articles(user_id, num_recommendations=5)
#print("Artigos recomendados:", recommended_article_ids)


