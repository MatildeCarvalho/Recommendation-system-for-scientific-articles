from tfidf_model import ArticleRecommendationSystemTFIDF
from entity_model import ArticleRecommendationSystem
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_dataframe(file_path):
    return pd.read_csv(file_path)

interactions_df = load_dataframe('file_final\\users_data.csv')

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

# Exemplo de uso:
recommendation_system_TFIDF = ArticleRecommendationSystemTFIDF('Train_teste_split\\train_set.csv', 'Train_teste_split\\test_set.csv', 'file_final\\articles_combined.csv')

#TFIDF
precision_1_tfidf = []
recall_1_tfidf = []
precision_2_tfidf = []
recall_2_tfidf = []
precision_3_tfidf = []
recall_3_tfidf = []
precision_4_tfidf = []
recall_4_tfidf = []
precision_5_tfidf = []
recall_5_tfidf = []

precision_values_tfidf = []
recall_values_tfidf = []

# Entity
precision_1_entity = []
recall_1_entity = []
precision_2_entity = []
recall_2_entity = []
precision_3_entity = []
recall_3_entity = []
precision_4_entity = []
recall_4_entity = []
precision_5_entity = []
recall_5_entity = []

for user_id in users:
    recommendation_system_TFIDF.train_user(user_id)

    # Obtenha os artigos que o usuário avaliou com rating 1 no conjunto de teste
    actual_items = recommendation_system_TFIDF.test[(recommendation_system_TFIDF.test['user'] == user_id) & (recommendation_system_TFIDF.test['rating'] == 1)]['article'].tolist()
    #print(f"Artigos avaliados pelo usuário {user_id}:', {actual_items}")

    #TF-IDF
    recommended_article_ids_1 = recommendation_system_TFIDF.recommend_articles(user_id, num_recommendations=1)
    recommended_article_ids_2 = recommendation_system_TFIDF.recommend_articles(user_id, num_recommendations=2)
    recommended_article_ids_3 = recommendation_system_TFIDF.recommend_articles(user_id, num_recommendations=3)
    recommended_article_ids_4 = recommendation_system_TFIDF.recommend_articles(user_id, num_recommendations=4)
    recommended_article_ids_5 = recommendation_system_TFIDF.recommend_articles(user_id, num_recommendations=5)
   

    # Calcule a precisão e o recall para o usuário atual
    precision_1 = precision_at_k(actual_items, recommended_article_ids_1, 1)
    recall_1 = recall_at_k(actual_items, recommended_article_ids_2, 1)
    precision_2 = precision_at_k(actual_items, recommended_article_ids_2, 2)
    recall_2 = recall_at_k(actual_items, recommended_article_ids_2, 2)
    precision_3 = precision_at_k(actual_items, recommended_article_ids_3, 3)
    recall_3 = recall_at_k(actual_items, recommended_article_ids_3, 3)
    precision_4 = precision_at_k(actual_items, recommended_article_ids_4, 4)
    recall_4 = recall_at_k(actual_items, recommended_article_ids_4, 4)
    precision_5 = precision_at_k(actual_items, recommended_article_ids_5, 5)
    recall_5 = recall_at_k(actual_items, recommended_article_ids_5, 5)

    # Store the Precision and Recall values for both TF-IDF and entities
    precision_1_tfidf.append(precision_1)
    recall_1_tfidf.append(recall_1)
    precision_2_tfidf.append(precision_2)
    recall_2_tfidf.append(recall_2)
    precision_3_tfidf.append(precision_3)
    recall_3_tfidf.append(recall_3)
    precision_4_tfidf.append(precision_4)
    recall_4_tfidf.append(recall_4)
    precision_5_tfidf.append(precision_5)
    recall_5_tfidf.append(recall_5)

    
    
    #print(f"User {user_id}: Precision@5 = {precision_5:.4f}, Recall@5 = {recall_5:.4f}")
    #print("Artigos recomendados:", recommended_article_ids)





# Calculate the average Precision and Recall values for both TF-IDF and entities
avg_precision_1_tfidf = np.mean(precision_1_tfidf)
avg_recall_1_tfidf = np.mean(recall_1_tfidf)
avg_precision_2_tfidf = np.mean(precision_2_tfidf)
avg_recall_2_tfidf = np.mean(recall_2_tfidf)
avg_precision_3_tfidf = np.mean(precision_3_tfidf)
avg_recall_3_tfidf = np.mean(recall_3_tfidf)
avg_precision_4_tfidf = np.mean(precision_4_tfidf)
avg_recall_4_tfidf = np.mean(recall_4_tfidf)
avg_precision_5_tfidf = np.mean(precision_5_tfidf)
avg_recall_5_tfidf = np.mean(recall_5_tfidf)

print("TF-IDF:")
print(f"Average Precision@1: {avg_precision_1_tfidf}")
print(f"Average Recall@1: {avg_recall_1_tfidf}")
print(f"Average Precision@2: {avg_precision_2_tfidf}")
print(f"Average Recall@2: {avg_recall_2_tfidf}")
print(f"Average Precision@3: {avg_precision_3_tfidf}")
print(f"Average Recall@3: {avg_recall_3_tfidf}")
print(f"Average Precision@4: {avg_precision_4_tfidf}")
print(f"Average Recall@4: {avg_recall_4_tfidf}")
print(f"Average Precision@5: {avg_precision_5_tfidf}")
print(f"Average Recall@5: {avg_recall_5_tfidf}")




recommendation_system_Entity = ArticleRecommendationSystem('Train_teste_split\\train_set.csv', 'Train_teste_split\\test_set.csv', 'file_final\\articles_combined.csv')
# Entity
precision_1_entity = []
recall_1_entity = []
precision_2_entity = []
recall_2_entity = []
precision_3_entity = []
recall_3_entity = []
precision_4_entity = []
recall_4_entity = []
precision_5_entity = []
recall_5_entity = []



for user_id in users:
    recommendation_system_Entity.train_user(user_id)

    # Obtenha os artigos que o usuário avaliou com rating 1 no conjunto de teste
    actual_items = recommendation_system_Entity.test[(recommendation_system_Entity.test['user'] == user_id) & (recommendation_system_Entity.test['rating'] == 1)]['article'].tolist()
    #print(f"Artigos avaliados pelo usuário {user_id}:', {actual_items}")

    #entity
    recommended_article_ids_1 = recommendation_system_Entity.recommend_articles(user_id, num_recommendations=1)
    recommended_article_ids_2 = recommendation_system_Entity.recommend_articles(user_id, num_recommendations=2)
    recommended_article_ids_3 = recommendation_system_Entity.recommend_articles(user_id, num_recommendations=3)
    recommended_article_ids_4 = recommendation_system_Entity.recommend_articles(user_id, num_recommendations=4)
    recommended_article_ids_5 = recommendation_system_Entity.recommend_articles(user_id, num_recommendations=5)
   

    # Calcule a precisão e o recall para o usuário atual
    precision_1 = precision_at_k(actual_items, recommended_article_ids_1, 1)
    recall_1 = recall_at_k(actual_items, recommended_article_ids_2, 1)
    precision_2 = precision_at_k(actual_items, recommended_article_ids_2, 2)
    recall_2 = recall_at_k(actual_items, recommended_article_ids_2, 2)
    precision_3 = precision_at_k(actual_items, recommended_article_ids_3, 3)
    recall_3 = recall_at_k(actual_items, recommended_article_ids_3, 3)
    precision_4 = precision_at_k(actual_items, recommended_article_ids_4, 4)
    recall_4 = recall_at_k(actual_items, recommended_article_ids_4, 4)
    precision_5 = precision_at_k(actual_items, recommended_article_ids_5, 5)
    recall_5 = recall_at_k(actual_items, recommended_article_ids_5, 5)

    # Store the Precision and Recall values for both TF-IDF and entities
    precision_1_entity.append(precision_1)
    recall_1_entity.append(recall_1)
    precision_2_entity.append(precision_2)
    recall_2_entity.append(recall_2)
    precision_3_entity.append(precision_3)
    recall_3_entity.append(recall_3)
    precision_4_entity.append(precision_4)
    recall_4_entity.append(recall_4)
    precision_5_tfidf.append(precision_5)
    recall_5_entity.append(recall_5)

    
    
    #print(f"User {user_id}: Precision@5 = {precision_5:.4f}, Recall@5 = {recall_5:.4f}")
    #print("Artigos recomendados:", recommended_article_ids)




# Calculate the average Precision and Recall values for both TF-IDF and entities
avg_precision_1_entity = np.mean(precision_1_entity)
avg_recall_1_entity = np.mean(recall_1_entity)
avg_precision_2_entity = np.mean(precision_2_entity)
avg_recall_2_entity = np.mean(recall_2_entity)
avg_precision_3_entity = np.mean(precision_3_entity)
avg_recall_3_entity = np.mean(recall_3_entity)
avg_precision_4_entity = np.mean(precision_4_entity)
avg_recall_4_entity = np.mean(recall_4_entity)
avg_precision_5_entity = np.mean(precision_5_entity)
avg_recall_5_entity = np.mean(recall_5_entity)

print("\nEntities:")
print(f"Average Precision@1: {avg_precision_1_entity}")
print(f"Average Recall@1: {avg_recall_1_entity}")
print(f"Average Precision@2: {avg_precision_2_entity}")
print(f"Average Recall@2: {avg_recall_2_entity}")
print(f"Average Precision@3: {avg_precision_3_entity}")
print(f"Average Recall@3: {avg_recall_3_entity}")
print(f"Average Precision@4: {avg_precision_4_entity}")
print(f"Average Recall@4: {avg_recall_4_entity}")
print(f"Average Precision@5: {avg_precision_5_entity}")
print(f"Average Recall@5: {avg_recall_5_entity}")

