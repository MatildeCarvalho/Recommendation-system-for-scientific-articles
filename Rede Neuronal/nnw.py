import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.nn as nn

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
data = pd.read_csv('Train_teste_split\\train_set.csv')
# Suponha que seu arquivo CSV tenha colunas 'user_id', 'article_id' e 'rating'
user_ids = torch.tensor(data['user'].values, dtype=torch.long)
article_ids = torch.tensor(data['article'].values, dtype=torch.long)
ratings = torch.tensor(data['rating'].values, dtype=torch.float)

# Crie uma instância do conjunto de dados
dataset = ArticleRecommendationDataset(user_ids, article_ids, ratings)

batch_size = 10
# Crie um DataLoader para carregar os dados em lotes
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
print(dataloader)

# Defina a classe do modelo de recomendação
class RecommendationModel(nn.Module):
    def __init__(self, num_users, num_articles, embedding_dim):
        super(RecommendationModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.article_embedding = nn.Embedding(num_articles, embedding_dim)
        self.fc = nn.Linear(embedding_dim * 2, 1)

    def forward(self, user_ids, article_ids):
        user_embeds = self.user_embedding(user_ids)
        article_embeds = self.article_embedding(article_ids)
        x = torch.cat([user_embeds, article_embeds], dim=1)
        output = self.fc(x)
        return output
    def monitor_metrics(self, outputs, ratings):
        # Calcule o erro quadrático médio
        loss = nn.MSELoss()
        mse = loss(outputs, ratings.unsqueeze(1).float())
        # Calcule a raiz quadrada do erro quadrático médio
        rmse = torch.sqrt(mse)
        return {'mse': mse.item(), 'rmse': rmse.item()}
    def predict(self, user_ids, article_ids):
        with torch.no_grad():
            outputs = model(user_ids, article_ids)
        return outputs.numpy()
    def save_model(self, path):
        torch.save(self.state_dict(), path)


# Defina hiperparâmetros
embedding_dim = 20  # Dimensão dos embeddings
num_epochs = 10  # Número de épocas de treinamento
learning_rate = 0.01  # Taxa de aprendizado

# Determine os valores máximos de user_ids e article_ids
max_user_id = user_ids.max().item()
max_article_id = article_ids.max().item()

# Defina o número máximo de usuários e artigos em suas camadas de incorporação
model = RecommendationModel(
    num_users=max_user_id + 1,  # Adicione 1 para incluir o ID máximo
    num_articles=max_article_id + 1,  # Adicione 1 para incluir o ID máximo
    embedding_dim=embedding_dim
)

# Função de treinamento
def train(model, dataloader, num_epochs, learning_rate):
    criterion = nn.MSELoss()  # Erro Quadrático Médio para regressão
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    # Listas para armazenar as métricas
    mse_values = []
    rmse_values = []
    
    for epoch in range(num_epochs):
        for batch in dataloader:
            user_ids = batch['user_id']
            article_ids = batch['article_id']
            ratings = batch['rating']

            optimizer.zero_grad()
            outputs = model(user_ids, article_ids)
            loss = criterion(outputs, ratings.unsqueeze(1).float())  # Converta ratings para tensor float
            loss.backward()
            optimizer.step()

        # Calcule as métricas para o conjunto de validação aqui (se aplicável)
        
        # Calcule as métricas de treinamento
        train_outputs = model(user_ids, article_ids)
        train_metrics = model.monitor_metrics(train_outputs, ratings)
        
        mse_values.append(train_metrics['mse'])
        rmse_values.append(train_metrics['rmse'])
        
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, MSE: {train_metrics["mse"]:.4f}, RMSE: {train_metrics["rmse"]:.4f}')

    
    return model, mse_values, rmse_values


# Treine o modelo
train(model, dataloader, num_epochs, learning_rate)

test_data = pd.read_csv('Train_teste_split\\test_set.csv')
# Suponha que seu arquivo CSV tenha colunas 'user_id', 'article_id' e 'rating'
test_user_ids = torch.tensor(test_data['user'].values, dtype=torch.long)
test_article_ids = torch.tensor(test_data['article'].values, dtype=torch.long)
test_ratings = torch.tensor(test_data['rating'].values, dtype=torch.float)

# Faça previsões no conjunto de teste
test_predictions = model.predict(test_user_ids, test_article_ids)
print(test_predictions)




# Epoch [1/10], Loss: 0.0000, MSE: 0.0000, RMSE: 0.0000
# Epoch [2/10], Loss: 0.0000, MSE: 0.0000, RMSE: 0.0000
# Epoch [3/10], Loss: 0.0000, MSE: 0.0000, RMSE: 0.0000





