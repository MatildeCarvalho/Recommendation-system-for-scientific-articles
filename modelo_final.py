### FINALLY, I GOT IT! ###


#import numpy as np

#data = [1.0, 2.0, np.nan, 4.0, 5.0]
#mean = np.nanmean(data)

#print(mean)  # Isso imprimirá '3.0', que é a média dos valores numéricos.


from tfidf_model import ArticleRecommendationSystemTFIDF
from entity_model import ArticleRecommendationSystem
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

# Check if CUDA (GPU) is available
if torch.cuda.is_available():
    # Set the device to GPU (0 or 1, etc.)
    device = torch.device("cuda:0")
else:
    # Use CPU if GPU is not available
    device = torch.device("cpu")

def load_dataframe(file_path):
    return pd.read_csv(file_path)

def calculate_metrics(recommendation_system, users, k_values):
    precision_values = {k: [] for k in k_values}
    recall_values = {k: [] for k in k_values}

    for user_id in users:
        recommendation_system.train_user(user_id)

        actual_items = recommendation_system.test[(recommendation_system.test['user'] == user_id) & (recommendation_system.test['rating'] == 1)]['article'].tolist()

        for k in k_values:
            recommended_article_ids = recommendation_system.recommend_articles(user_id, num_recommendations=k)

            if k > len(actual_items):
                precision = recall = np.nan
            else:
                num_correct = len(set(actual_items) & set(recommended_article_ids))
                precision = num_correct / k if k != 0 else 0
                recall = num_correct / len(actual_items) if len(actual_items) != 0 else 0 

            precision_values[k].append(precision)
            recall_values[k].append(recall)

    avg_precision = {k: np.nanmean(values) for k, values in precision_values.items()}
    avg_recall = {k: np.nanmean(values) for k, values in recall_values.items()}

    return avg_precision, avg_recall

interactions_df = load_dataframe('file_final\\users_data.csv')
users = interactions_df['user'].unique()

recommendation_system_TFIDF = ArticleRecommendationSystemTFIDF('Train_teste_split\\train_set.csv', 'Train_teste_split\\test_set.csv', 'file_final\\articles_combined.csv')
recommendation_system_Entity = ArticleRecommendationSystem('Train_teste_split\\train_set.csv', 'Train_teste_split\\test_set.csv', 'file_final\\articles_combined.csv')


k_values = [1, 2, 3, 4, 5]

avg_precision_tfidf, avg_recall_tfidf = calculate_metrics(recommendation_system_TFIDF, users, k_values)
avg_precision_entity, avg_recall_entity = calculate_metrics(recommendation_system_Entity, users, k_values)

print("TF-IDF Metrics:")
for k, precision in avg_precision_tfidf.items():
    print(f"Average Precision@{k}: {precision:.4f}")
for k, recall in avg_recall_tfidf.items():
    print(f"Average Recall@{k}: {recall:.4f}")

print("\nEntity Metrics:")
for k, precision in avg_precision_entity.items():
    print(f"Average Precision@{k}: {precision:.4f}")
for k, recall in avg_recall_entity.items():
    print(f"Average Recall@{k}: {recall:.4f}")

# Plot Recall vs. K
plt.figure(figsize=(10, 5))
plt.plot(k_values, [round(avg_recall_tfidf[k], 4) for k in k_values], label='TF-IDF', marker='o')
plt.plot(k_values, [round(avg_recall_entity[k], 4) for k in k_values], label='Entity', marker='o')
plt.title('Recall vs. K')
plt.xlabel('K')
plt.ylabel('Recall')
plt.legend()
plt.grid(True)
plt.savefig('results//recall_vs_k.png')  # Salva o gráfico como um arquivo PNG
plt.show()

# Plot Precision vs. K
plt.figure(figsize=(10, 5))
plt.plot(k_values, [round(avg_precision_tfidf[k], 4) for k in k_values], label='TF-IDF', marker='o')
plt.plot(k_values, [round(avg_precision_entity[k], 4) for k in k_values], label='Entity', marker='o')
plt.title('Precision vs. K')
plt.xlabel('K')
plt.ylabel('Precision')
plt.legend()
plt.grid(True)
plt.savefig('results//precision_vs_k.png')  # Salva o gráfico como um arquivo PNG
plt.show()


# Dados de precisão e recall
final_results = {
    'K': k_values,
    'Precision (TF-IDF)': [round(avg_precision_tfidf[k], 4) for k in k_values],
    'Precision (Entity)': [round(avg_precision_entity[k], 4) for k in k_values],
    'Recall (TF-IDF)': [round(avg_recall_tfidf[k], 4) for k in k_values],
    'Recall (Entity)': [round(avg_recall_entity[k], 4) for k in k_values],
}

# Crie um DataFrame com os dados
results = pd.DataFrame(final_results)

results.to_csv('results//resultado_final.csv', index=False)


#TF-IDF Metrics:
#Average Precision@1: 0.5815
#Average Precision@2: 0.5691
#Average Precision@3: 0.5740
#Average Precision@4: 0.5760
#Average Precision@5: 0.5810
#Average Recall@1: 0.1534
#Average Recall@2: 0.3007
#Average Recall@3: 0.3148
#Average Recall@4: 0.3228
#Average Recall@5: 0.3324

#Entity Metrics:
#Average Precision@1: 1.0000
#Average Precision@2: 0.9996
#Average Precision@3: 0.9998
#Average Precision@4: 0.9999
#Average Precision@5: 1.0000
#Average Recall@1: 0.2738
#Average Recall@2: 0.5473
#Average Recall@3: 0.5676
#Average Recall@4: 0.5814
#Average Recall@5: 0.5927
