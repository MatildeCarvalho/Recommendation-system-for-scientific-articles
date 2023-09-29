import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.metrics import mean_squared_error
import math

import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Defina a classe do conjunto de dados
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
        return {'user_id': user_id, 'article_id': article_id, 'rating': rating}

# Carregue seu arquivo CSV em um DataFrame do Pandas
data = pd.read_csv('Train_teste_split/train_set.csv')
# Suponha que seu arquivo CSV tenha colunas 'user_id', 'article_id' e 'rating'
user_ids = torch.tensor(data['user'].values, dtype=torch.long)
article_ids = torch.tensor(data['article'].values, dtype=torch.long)
ratings = torch.tensor(data['rating'].values, dtype=torch.float)

# Crie uma instância do conjunto de dados
dataset = ArticleRecommendationDataset(user_ids, article_ids, ratings)

batch_size = 10
# Crie um DataLoader para carregar os dados em lotes
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Função para calcular RMSE e MSE
def calculate_metrics(predictions, targets):
    mse = mean_squared_error(targets, predictions)
    rmse = math.sqrt(mse)
    return mse, rmse

# Função de treinamento
def train_model(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for batch in dataloader:
        user_ids_batch = batch['user_id'].to(device)
        article_ids_batch = batch['article_id'].to(device)
        ratings_batch = batch['rating'].to(device)

        # Forward pass
        outputs = model(user_ids_batch, article_ids_batch)
        loss = criterion(outputs.squeeze(), ratings_batch)

        # Backward pass e otimização
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Retorna a perda média durante o treinamento
    return running_loss / len(dataloader)

# Função de teste
def test_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            user_ids_batch = batch['user_id'].to(device)
            article_ids_batch = batch['article_id'].to(device)
            ratings_batch = batch['rating'].to(device)

            # Forward pass
            outputs = model(user_ids_batch, article_ids_batch)
            loss = criterion(outputs.squeeze(), ratings_batch)
            running_loss += loss.item()

            all_predictions.extend(outputs.squeeze().cpu().numpy())
            all_targets.extend(ratings_batch.cpu().numpy())

    # Calcula as métricas RMSE e MSE
    mse, rmse = calculate_metrics(all_predictions, all_targets)

    # Retorna a perda média e as métricas RMSE e MSE durante o teste
    return running_loss / len(dataloader), mse, rmse

# Defina o modelo de recomendação
class MovieRecommendationModel(nn.Module):
    def __init__(self, num_users, num_articles, embedding_dim):
        super(MovieRecommendationModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.article_embedding = nn.Embedding(num_articles, embedding_dim)
        self.fc = nn.Linear(embedding_dim * 2, 1)

    def forward(self, user_ids, article_ids):
        user_embeds = self.user_embedding(user_ids)
        article_embeds = self.article_embedding(article_ids)
        concat_embeds = torch.cat((user_embeds, article_embeds), dim=1)
        rating = self.fc(concat_embeds)
        return rating

# Hiperparâmetros
embedding_dim = 32
num_epochs = 10
learning_rate = 0.001

# Determine os valores máximos de user_ids e article_ids
max_user_id = user_ids.max().item()
max_article_id = article_ids.max().item()

# Defina o número máximo de usuários e artigos em suas camadas de incorporação
model = MovieRecommendationModel(
    num_users=max_user_id + 1,
    num_articles=max_article_id + 1,
    embedding_dim=embedding_dim
)

# Mova o modelo para o dispositivo (GPU ou CPU)
model.to(device)

# Defina a função de perda e otimizador
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Treinamento do modelo
for epoch in range(num_epochs):
    train_loss = train_model(model, dataloader, criterion, optimizer, device)
    test_loss, test_mse, test_rmse = test_model(model, dataloader, criterion, device)

    print(f'Epoch [{epoch+1}/{num_epochs}]')
    print(f'Train Loss: {train_loss:.4f}')
    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test MSE: {test_mse:.4f}, Test RMSE: {test_rmse:.4f}')

# Salvando o modelo treinado
torch.save(model.state_dict(), 'Rede Neuronal/movie_recommendation_model.pth')


# Epoch [1/10], Loss: 3.309028397779912e-05
# Epoch [2/10], Loss: 0.000337200122885406
# Epoch [3/10], Loss: 2.2563463062397204e-05
# Epoch [4/10], Loss: 6.2964027165435255e-06
# Epoch [5/10], Loss: 0.00023356823658104986
# Epoch [6/10], Loss: 1.1933348105230834e-05
# Epoch [7/10], Loss: 3.496700810501352e-05
# Epoch [8/10], Loss: 0.00010050585842691362
# Epoch [9/10], Loss: 4.803114279638976e-06
# Epoch [10/10], Loss: 6.261906946747331e-06




