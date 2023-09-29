## https://www.youtube.com/watch?v=Wj-nkk7dFS8&t=1257s



import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import math
import matplotlib.pyplot as plt
import logging
from sklearn import model_selection
from sklearn.metrics import precision_score, recall_score
import numpy as np

# Constants for hyperparameters
EMBEDDING_DIM = 4
LR = 1e-3
WD = 0.0
BATCH_SIZE = 5
EPOCHS = 5
K = 5 # Valor de k para precisão@k e recall@k

# Early stopping parameters
PATIENCE = 2  # Number of epochs with no improvement to wait before early stopping

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

class ArticleRecommendationDataset(Dataset):
    def __init__(self, user_ids, article_ids, ratings):
        self.user_ids = user_ids
        self.article_ids = article_ids
        self.ratings = ratings

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        article_id = self.article_ids[idx]
        rating = self.ratings[idx]
        return {'user_id': torch.tensor(user_id, dtype=torch.long), 
                'article_id': torch.tensor(article_id, dtype=torch.long), 
                'rating': torch.tensor(rating, dtype=torch.float)}

class RecommendationModel(nn.Module):
    def __init__(self, num_users, num_articles, embedding_dim=32):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.article_embedding = nn.Embedding(num_articles, embedding_dim)
        self.fc = nn.Linear(embedding_dim * 2, 1)
        self.step_scheduler_after = "epoch"
    
    def fetch_optimizer(self, lr, wd):
        opt = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)
        return opt

    def fetch_scheduler(self, optimizer):
        sch = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)  # Adjust step_size and gamma as needed
        return sch

    def forward(self, users, articles, ratings):
        user_embeds = self.user_embedding(users)
        article_embeds = self.article_embedding(articles)
        x = torch.cat([user_embeds, article_embeds], dim=1)
        output = self.fc(x)
        loss = nn.MSELoss()(output, ratings.unsqueeze(1).float())
        return output, loss

def plot_losses(train_losses, valid_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', marker='o')
    plt.plot(valid_losses, label='Valid Loss', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.grid(True)
    plt.savefig('Rede Neuronal/loss_plot.png')
    plt.show()

def train(model, train_loader, valid_loader, epochs, lr, wd, save_path):
    optimizer = model.fetch_optimizer(lr, wd)
    scheduler = model.fetch_scheduler(optimizer)

    best_valid_loss = float('inf')
    best_model_state_dict = None
    train_losses = []  # To store training losses
    valid_losses = []  # To store validation losses
    patience_counter = 0  # Counter for early stopping

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            users, articles, ratings = batch['user_id'].to(device), batch['article_id'].to(device), batch['rating'].to(device)
            optimizer.zero_grad()
            outputs, loss = model(users, articles, ratings)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        # Adicione a linha abaixo para atualizar o scheduler
        scheduler.step()
        model.eval()
        valid_loss = 0.0

        with torch.no_grad():
            for batch in valid_loader:
                users, articles, ratings = batch['user_id'].to(device), batch['article_id'].to(device), batch['rating'].to(device)
                outputs, loss = model(users, articles, ratings)
                valid_loss += loss.item()

            valid_loss /= len(valid_loader)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_model_state_dict = model.state_dict()
                torch.save(best_model_state_dict, save_path)
                patience_counter = 0  # Reset patience counter
            else:
                patience_counter += 1  # Increment patience counter

            train_losses.append(train_loss)
            valid_losses.append(valid_loss)

            logger.info(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f} - Valid Loss: {valid_loss:.4f}")

            if patience_counter >= PATIENCE:
                logger.info(f"Early stopping after {PATIENCE} epochs with no improvement.")
                break

    plot_losses(train_losses, valid_losses)  # Plot losses for each epoch

    return best_model_state_dict

if __name__ == "__main__":
    df = pd.read_csv('Train_teste_split/train_set.csv') # user, article, rating
    lbl_enc_user = LabelEncoder()
    lbl_enc_article = LabelEncoder()
    df.user = lbl_enc_user.fit_transform(df.user.values) # encode user and article IDs to start from 0
    df.article = lbl_enc_article.fit_transform(df.article.values)

    df_train, df_valid = model_selection.train_test_split(df, test_size=0.1, random_state=42, stratify=df.rating.values)
    train_dataset = ArticleRecommendationDataset(df_train.user.values, df_train.article.values, df_train.rating.values)
    valid_dataset = ArticleRecommendationDataset(df_valid.user.values, df_valid.article.values, df_valid.rating.values)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = RecommendationModel(num_users=len(lbl_enc_user.classes_), num_articles=len(lbl_enc_article.classes_), embedding_dim=EMBEDDING_DIM).to(device)
    
    best_model_state_dict = train(model, train_loader, valid_loader, epochs=EPOCHS, lr=LR, wd=WD, save_path='Rede Neuronal/model.pth')

    # Agora que o treinamento terminou, você pode carregar o modelo treinado e realizar o teste
    test_df = pd.read_csv('Train_teste_split/test_set.csv')  # Carregue o conjunto de dados de teste
    lbl_enc_user_test = LabelEncoder()
    lbl_enc_article_test = LabelEncoder()
    test_df.user = lbl_enc_user_test.fit_transform(test_df.user.values) # encode user and article IDs to start from 0
    test_df.article = lbl_enc_article_test.fit_transform(test_df.article.values)

    test_dataset = ArticleRecommendationDataset(test_df.user.values, test_df.article.values, test_df.rating.values)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Certifique-se de que o modelo esteja no modo de avaliação
    model.eval()

    # ... (código para realizar inferências no conjunto de dados de teste e calcular métricas)

    # Exiba ou salve as métricas, conforme necessário
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








#def load_data():
# https://www.youtube.com/watch?v=zThs-4EtJdA
#    pass

