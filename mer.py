import pandas as pd
#from lenskit import crossfold
import numpy as np

from sklearn.preprocessing import MultiLabelBinarizer
import random
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

def load_dataframe(file_path):
    return pd.read_csv(file_path)

articles_df = load_dataframe('file_final\\articles_combined.csv')
interactions_df = load_dataframe('file_final\\users_data.csv')



class TFIDFModel:
    def __init__(self, articles_df):
        self.abstract_matrix = self.get_article_abstract_matrix(articles_df)

    def get_article_abstract_matrix(self, articles_df):
        abstracts = articles_df['clean_abstract'].tolist()

        tfidf = TfidfVectorizer()
        abstract_matrix = pd.DataFrame(tfidf.fit_transform(abstracts).toarray(), columns=tfidf.get_feature_names_out(), index=articles_df['doc.id'])

        return abstract_matrix

    def get_user_profile(self, user_ratings):
        articles = user_ratings['article'].tolist()
        user_abstract_matrix = self.abstract_matrix.loc[articles]

        user_profile = user_abstract_matrix.mean(axis=0)
        user_profile_normalized = user_profile / np.linalg.norm(user_profile)

        return user_profile_normalized


class EntityModel:
    def __init__(self, articles_df):
        self.genre_matrix = self.get_movie_genre_matrix(articles_df)

    def get_movie_genre_matrix(self, articles_df):
        genres = articles_df['entities'].apply(eval).tolist()

        mlb = MultiLabelBinarizer()
        genre_matrix = pd.DataFrame(mlb.fit_transform(genres), columns=mlb.classes_, index=articles_df['doc.id'])

        return genre_matrix

    def get_user_profile(self, user_ratings):
        articles = user_ratings['article'].tolist()
        user_genre_matrix = self.genre_matrix.loc[articles]

        user_profile = user_genre_matrix.mean(axis=0)
        user_profile_normalized = user_profile / np.linalg.norm(user_profile)

        return user_profile_normalized


def create_article_sample(interactions_df, user_id, num_samples):
    # Obter todos os valores únicos da coluna 'article'
    all_articles = interactions_df['article'].unique()

    # Filtrar os artigos que o user já leu
    user_articles = interactions_df.loc[interactions_df['user'] == user_id, 'article'].unique()

    # Obter os artigos que o user ainda não leu
    articles_not_read = np.setdiff1d(all_articles, user_articles)

    # Amostra aleatória dos artigos não lidos
    article_sample = np.random.choice(articles_not_read, size=num_samples, replace=False)

    # Criar um DataFrame com os artigos da amostra
    sample_df = pd.DataFrame({
        'user': user_id,
        'article': article_sample,
        'rating': 1
    })

    return sample_df

def get_user_ratings(interactions_df, user_id):
    user_ratings = interactions_df.loc[interactions_df['user'] == user_id, ['article', 'rating']]
    return user_ratings


def get_user_genre_matrix(user_ratings, genre_matrix):
    articles = user_ratings['article'].tolist()
    user_genre_matrix = genre_matrix.loc[articles]

    return user_genre_matrix


def get_user_profile(user_genre_matrix):
    user_profile = user_genre_matrix.mean(axis=0)
    user_profile_normalized = user_profile / np.linalg.norm(user_profile)

    return user_profile_normalized

def select_test_rows(df, test_percentage=0.2):
    # Calcula o tamanho do conjunto de teste
    test_size = int(len(df) * test_percentage)

    # Embaralha o dataframe para seleção aleatória
    df_shuffled = df.sample(frac=1, random_state=2)

    # Divide o dataframe em conjunto de teste e treinamento
    test_rows = df_shuffled.head(test_size)
    train_rows = df_shuffled.tail(len(df_shuffled) - test_size)

    return train_rows, test_rows

def get_top_n_recommendations(recommendation_scores, n):
    top_n_articles = recommendation_scores.nlargest(n)
    return top_n_articles.index.tolist()

def precision_at_k(actual_items, predicted_items, k):


    # Obter os itens preditos no top-K
    predicted_items_k = predicted_items[:k]

    # Calcular o número de itens corretamente preditos
    num_correct = len(set(actual_items) & set(predicted_items_k))

    # Calcular o Precision@K
    precision = num_correct / k

    return precision


def recall_at_k(actual_items, predicted_items, k):

    # Obter os itens preditos no top-K
    predicted_items_k = predicted_items[:k]

    # Calcular o número de itens corretamente preditos
    num_correct = len(set(actual_items) & set(predicted_items_k))

    # Calcular o Recall@K
    recall = num_correct / len(actual_items)

    return recall





# Supondo que você tenha um dataframe chamado "interactions_df" com todas as interações dos users
users = interactions_df['user'].unique()


# Initialize the models for both TF-IDF and entities
tfidf_model = TFIDFModel(articles_df)
entity_model = EntityModel(articles_df)

precision_values_tfidf = []
recall_values_tfidf = []

precision_values_entity = []
recall_values_entity = []


for user in users:
    user_rows = interactions_df.loc[interactions_df['user'] == user]

    # Split the user rows into train and test sets per user
    train_rows, test_rows = select_test_rows(user_rows, test_percentage=0.2)

    # Create the sample for the user
    sample_df = create_article_sample(interactions_df, user, num_samples=2*len(test_rows))

    # Append the sample to the test set
    test_rows_with_sample = pd.concat([test_rows, sample_df])

    # Vector with the ratings of the user for the test set and train set
    user_ratings_train = get_user_ratings(train_rows, user)
    user_ratings_test = get_user_ratings(test_rows_with_sample, user)

    # Get the user-item matrix for both TF-IDF and entities in train set
    user_item_matrix_tfidf_train = get_user_genre_matrix(user_ratings_train, tfidf_model.abstract_matrix)
    user_item_matrix_entity_train = get_user_genre_matrix(user_ratings_train, entity_model.genre_matrix)

    # User profile for both TF-IDF and entities
    user_profile_tfidf = get_user_profile(user_item_matrix_tfidf_train)
    user_profile_entity = get_user_profile(user_item_matrix_entity_train)

    # Get the user-item matrix for both TF-IDF and entities in test set
    user_item_matrix_tfidf_test = get_user_genre_matrix(user_ratings_test, tfidf_model.abstract_matrix)
    user_item_matrix_entity_test = get_user_genre_matrix(user_ratings_test, entity_model.genre_matrix)

    # Calculate the recommendation scores for both TF-IDF and entities
    recommendation_scores_tfidf = np.sum(user_item_matrix_tfidf_test * user_profile_tfidf, axis=1)
    recommendation_scores_entity = np.sum(user_item_matrix_entity_test * user_profile_entity, axis=1)

    min_score_tfidf = np.min(recommendation_scores_tfidf)
    max_score_tfidf = np.max(recommendation_scores_tfidf)
    normalized_scores_tfidf = (recommendation_scores_tfidf - min_score_tfidf) / (max_score_tfidf - min_score_tfidf)

    min_score_entity = np.min(recommendation_scores_entity)
    max_score_entity = np.max(recommendation_scores_entity)
    normalized_scores_entity = (recommendation_scores_entity - min_score_entity) / (max_score_entity - min_score_entity)

    # Get the top N recommendations for both TF-IDF and entities
    top_n_articles_tfidf = get_top_n_recommendations(normalized_scores_tfidf, 3)
    top_n_articles_entity = get_top_n_recommendations(normalized_scores_entity, 3)

    # Calculate Precision@K and Recall@K for both TF-IDF and entities
    if len(test_rows['article'].tolist()) >= 3:
        precision_tfidf = precision_at_k(test_rows['article'].tolist(), top_n_articles_tfidf, k=3)
        recall_tfidf = recall_at_k(test_rows['article'].tolist(), top_n_articles_tfidf, k=3)

        precision_entity = precision_at_k(test_rows['article'].tolist(), top_n_articles_entity, k=3)
        recall_entity = recall_at_k(test_rows['article'].tolist(), top_n_articles_entity, k=3)

    # Store the Precision and Recall values for both TF-IDF and entities
    precision_values_tfidf.append(precision_tfidf)
    recall_values_tfidf.append(recall_tfidf)

    precision_values_entity.append(precision_entity)
    recall_values_entity.append(recall_entity)

# Calculate the average Precision and Recall values for both TF-IDF and entities
avg_precision_tfidf = np.mean(precision_values_tfidf)
avg_recall_tfidf = np.mean(recall_values_tfidf)

avg_precision_entity = np.mean(precision_values_entity)
avg_recall_entity = np.mean(recall_values_entity)

print("TF-IDF:")
print(f"Average Precision@3: {avg_precision_tfidf}")
print(f"Average Recall@3: {avg_recall_tfidf}")

print("\nEntities:")
print(f"Average Precision@3: {avg_precision_entity}")
print(f"Average Recall@3: {avg_recall_entity}")