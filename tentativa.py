
# funciona 

import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import MultiLabelBinarizer


def load_dataframe(file_path):
    return pd.read_csv(file_path)

train = load_dataframe('Train_teste_split\\train_set.csv')
test = load_dataframe('Train_teste_split\\test_set.csv')
articles_df = load_dataframe('file_final\\articles_combined.csv')


def get_user_ratings(user_id, train):
    """
    Retorna os IDs dos filmes e os ratings que um usuário deu aos artigos no train_set.

    Args:
        user_id (int): O ID do usuário.
        train (pd.DataFrame): O DataFrame contendo o conjunto de treinamento com as colunas 'user', 'article' e 'rating'.

    Returns:
        tuple: Uma tupla contendo dois elementos:
            - Uma lista de IDs de filmes que o usuário avaliou.
            - Uma lista de ratings que o usuário deu aos filmes correspondentes.
    """
    user_data = train[train['user'] == user_id]
    articles_ids = user_data['article'].tolist()
    user_ratings = user_data['rating'].tolist()
    return articles_ids, user_ratings


def get_entities_for_articles(article_ids, articles_df):
    """
    Retorna uma lista de listas de entidades correspondentes a cada artigo na lista de IDs fornecida.

    Args:
        article_ids (list): Uma lista de IDs de artigos.
        articles_df (pd.DataFrame): O DataFrame contendo as colunas 'doc.id' e 'entities'.

    Returns:
        list: Uma lista de listas de entidades correspondentes a cada artigo na lista de IDs fornecida.
    """
    entities_lists = []

    for article_id in article_ids:
        article_data = articles_df[articles_df['doc.id'] == article_id]
        entities_sequence = article_data['entities'].values[0]
        entities_list = re.findall(r"'([^']+)'", entities_sequence)
        entities_lists.append(entities_list)

    return entities_lists



def get_all_user_entities(user_id, train_set, test_set):
    """
    Retorna uma lista de entidades únicas do train e do test do usuário.

    Args:
        user_id (int): O ID do usuário.
        train_set (pd.DataFrame): O DataFrame contendo o conjunto de treinamento com as colunas 'user' e 'entities'.
        test_set (pd.DataFrame): O DataFrame contendo o conjunto de teste com as colunas 'user' e 'entities'.

    Returns:
        list: Uma lista de entidades únicas do train e do test do usuário.
    """
    user_entities = pd.concat([train_set, test_set], axis=0)
    user_entities = user_entities[user_entities['user'] == user_id]['entities']
    unique_entities = set()
    
    for sequence in user_entities:
        entities = re.findall(r"'([^']+)'", sequence)
        unique_entities.update(entities)

    return list(unique_entities)


def criar_matriz_articles_entidades(articles, entidades_unicas, sequencias_entidades):
    """
    Cria uma matriz que representa a presença e contagem de entidades em cada filme.

    Args:
        articles (list): Uma lista de artigos (ou filmes).
        entidades_unicas (list): Uma lista de nomes de entidades únicas.
        sequencias_entidades (list): Uma lista de listas de entidades para cada filme (ou artigo).

    Returns:
        np.ndarray: Uma matriz onde cada linha representa um filme (ou artigo) e cada coluna representa uma entidade.
                    Os valores nas células indicam o número de ocorrências da entidade no filme.

    Exemplo:
        Suponha:
        articles = ["Artigo1", "Artigo2", "Artigo3"]
        entidades_unicas = ["Entidade1", "Entidade2", "Entidade3", "Entidade4"]
        sequencias_entidades = [
            ["Entidade1", "Entidade3"],
            ["Entidade2", "Entidade4"],
            ["Entidade1", "Entidade2", "Entidade1"]  # Artigo 3 tem duas vezes a Entidade1
        ]

        A chamada da função:
        matriz_articles_entidades = criar_matriz_articles_entidades(articles, entidades_unicas, sequencias_entidades)

        Resultará em uma matriz NumPy como esta:
        array([[1, 0, 1, 0],
               [0, 1, 0, 1],
               [2, 1, 0, 0]])
    """
    # Inicialize a matriz com zeros
    matriz_articles_entidades = np.zeros((len(articles), len(entidades_unicas)), dtype=int)

    # Preencha a matriz com a contagem de ocorrências das entidades nas sequências
    for i, article in enumerate(articles):
        for j, entidade in enumerate(entidades_unicas):
            matriz_articles_entidades[i][j] = sequencias_entidades[i].count(entidade)

    return matriz_articles_entidades




def weighted_entity_matrix(articles_matrix, user_ratings):
    """
    Retorna uma matriz de entidades ponderadas com base nos ratings do usuário.

    Args:
        articles_matrix (list): Uma matriz onde cada linha representa um filme e cada coluna representa uma entidade.
                                Os valores nas células indicam o número de ocorrências da entidade no filme.
        user_ratings (list): Uma lista de ratings que o usuário deu aos filmes correspondentes.

    Returns:
        np.ndarray: Uma matriz NumPy onde cada linha representa um filme e cada coluna representa uma entidade.
                    Os valores nas células indicam a ponderação da entidade no filme.
    """
    # Converta as listas para arrays NumPy para que a multiplicação seja mais eficiente
    articles_matrix = np.array(articles_matrix)
    user_ratings = np.array(user_ratings)

    # Realize a multiplicação elemento a elemento
    weighted_matrix = articles_matrix * user_ratings[:, np.newaxis]

    return weighted_matrix

def user_profile(weighted_matrix):
    """
    Retorna o perfil do usuário normalizado (norma L2 do vetor -- distância Euclidiana do vetor a partir da origem (0, 0) em um espaço multidimensional).

    Args:
        weighted_matrix (np.ndarray): Uma matriz NumPy onde cada linha representa um filme e cada coluna representa uma entidade.
                                      Os valores nas células indicam a ponderação da entidade no filme.

    Returns:
        np.ndarray: Um array NumPy que representa o perfil do usuário normalizado.
    """
    # Calcule o perfil do usuário somando as linhas da matriz ponderada
    user_profile = np.sum(weighted_matrix, axis=0)
    
    # Calcule a norma L2 do vetor do perfil do usuário
    norm = np.linalg.norm(user_profile)
    
    # Evite a divisão por zero se o vetor for um vetor nulo
    if norm == 0:
        return user_profile
    
    # Normalize o vetor do perfil do usuário dividindo cada elemento pelo valor da norma
    normalized_user_profile = user_profile / norm

    return normalized_user_profile

def weighted_articles_matrix(test_articles_matrix, user_profile):
    """
    Retorna a matriz ponderada das entidades dos artigos de teste com base no perfil do usuário.

    Args:
        test_articles_matrix (np.ndarray): Uma matriz NumPy onde cada linha representa um artigo de teste e cada coluna
                                           representa uma entidade. Os valores nas células indicam o número de ocorrências
                                           da entidade no artigo de teste.
        user_profile (np.ndarray): Um vetor NumPy que representa o perfil do usuário normalizado.

    Returns:
        np.ndarray: Uma matriz onde cada linha representa um artigo de teste e cada coluna representa uma entidade.
                    Os valores nas células indicam a ponderação da entidade no artigo de teste com base no perfil do usuário.
    """
    # Realize a multiplicação da matriz de testes pelo perfil do usuário elemento a elemento
    weighted_matrix = test_articles_matrix * user_profile

    return weighted_matrix

from sklearn.metrics.pairwise import cosine_similarity

from sklearn.metrics.pairwise import cosine_similarity

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def recommend_articles(test_articles_matrix, user_profile, article_ids, num_recommendations=5):
    """
    Recomenda artigos ao usuário com base na semelhança do cosseno entre o perfil do usuário e os artigos de teste.

    Args:
        test_articles_matrix (np.ndarray): Uma matriz NumPy onde cada linha representa um artigo de teste e cada coluna
                                           representa uma entidade. Os valores nas células indicam a ponderação da entidade
                                           no artigo de teste.
        user_profile (np.ndarray): Um vetor NumPy que representa o perfil do usuário normalizado.
        article_ids (list): Uma lista de IDs dos artigos correspondentes aos artigos de teste.
        num_recommendations (int): O número de artigos a serem recomendados (padrão: 5).

    Returns:
        list: Uma lista de IDs dos artigos recomendados.
    """
    # Calcule a semelhança do cosseno entre o perfil do usuário e os artigos de teste
    similarities = cosine_similarity([user_profile], test_articles_matrix)

    # Obtém os índices dos artigos recomendados com base na semelhança do cosseno
    recommended_indices = similarities.argsort()[0][::-1]

    # Seleciona os IDs dos artigos recomendados
    recommended_article_ids = [article_ids[i] for i in recommended_indices[:num_recommendations]]

    return recommended_article_ids


#TRAIN
# Exemplo de uso:
articles_ids_list, user_ratings = get_user_ratings(0, train)
entities_for_articles = get_entities_for_articles(articles_ids_list, articles_df)
#print(entities_for_articles)
articles_matrix=criar_matriz_articles_entidades(articles_ids_list, get_all_user_entities(0, train, test), entities_for_articles)
weighted_matrix = weighted_entity_matrix(articles_matrix, user_ratings)
#print(weighted_matrix)
user_profiles = user_profile(weighted_matrix)
print(user_profiles.shape)
#print(len(user_profiles))
#print(len(articles_ids_list))
#print(weighted_matrix.shape)

#TEST
# Exemplo de uso:
articles_ids_list_TEST, user_ratings = get_user_ratings(0, test)
entities_for_articles_TEST = get_entities_for_articles(articles_ids_list_TEST, articles_df)
#print(entities_for_articles)
candidate_articles_matrix=criar_matriz_articles_entidades(articles_ids_list_TEST, get_all_user_entities(0, train, test), entities_for_articles_TEST)
test_articles_matrix = weighted_entity_matrix(candidate_articles_matrix, user_ratings)

# Exemplo de uso:
# Suponha que você tenha a matriz de testes 'test_articles_matrix' e o perfil do usuário 'user_profile'
# Substitua esses valores pelos seus dados reais
recommended_article_indices = recommend_articles(test_articles_matrix, user_profiles, articles_ids_list_TEST, num_recommendations=5)
print(recommended_article_indices)