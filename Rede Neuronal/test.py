import torch
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score
from train import RecommendationModel, ArticleRecommendationDataset  # Importe a classe do modelo e o Dataset
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

# Constants
BATCH_SIZE = 5  # Use o mesmo BATCH_SIZE que você usou durante o treinamento
MODEL_PATH = 'Rede Neuronal/model.pth'  # Caminho para o modelo treinado
TEST_CSV_PATH = 'Train_teste_split/test_set.csv'  # Caminho para o arquivo CSV de teste
K = 5  # Valor de k para precisão@k e recall@k
EMBEDDING_DIM = 4  # Certifique-se de fornecer a mesma dimensão que usou durante o treinamento

if __name__ == "__main__":
    # Carregue o conjunto de dados de teste
    df = pd.read_csv(TEST_CSV_PATH) # user, article, rating
    lbl_enc_user = LabelEncoder()
    lbl_enc_article = LabelEncoder()
    df.user = lbl_enc_user.fit_transform(df.user.values) # encode user and article IDs to start from 0
    df.article = lbl_enc_article.fit_transform(df.article.values)

    test_dataset = ArticleRecommendationDataset(df.user.values, df.article.values, df.rating.values)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Carregue o modelo treinado
    model = RecommendationModel(num_users=len(lbl_enc_user.classes_), num_articles=len(lbl_enc_article.classes_), embedding_dim=EMBEDDING_DIM)
    model.load_state_dict(torch.load(MODEL_PATH))

    # Certifique-se de definir o modelo para o modo de avaliação (isso desativará camadas como Dropout)
    model.eval()

    # Realize inferências no conjunto de dados de teste
    all_predictions = []
    all_actuals = []

    with torch.no_grad():
        for batch in test_loader:
            users, articles, ratings = batch['user_id'], batch['article_id'], batch['rating']
            outputs, _ = model(users, articles, ratings)  # Não precisa da perda ou métricas aqui
            predictions = outputs.squeeze().cpu().numpy()  # Converta para numpy

            all_predictions.extend(predictions)
            all_actuals.extend(ratings.numpy())  # Use os valores reais de classificação do lote

    # Ordene as previsões e reais
    sorted_indices = np.argsort(all_predictions, axis=0)[::-1]
    sorted_actuals = np.array(all_actuals)[sorted_indices]

    # Calcule precisão@k e recall@k
    top_k_actuals = sorted_actuals[:K]
    precision_at_k = precision_score(np.ones(K), top_k_actuals)  # np.ones(K) representa os itens recomendados
    recall_at_k = recall_score(np.ones(K), top_k_actuals)  # np.ones(K) representa os itens recomendados

    print(f'Precisão@{K}: {precision_at_k:.4f}')
    print(f'Recall@{K}: {recall_at_k:.4f}')



    
