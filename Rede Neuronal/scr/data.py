import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Verifique a disponibilidade de GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carregue seus dados (substitua pelo carregamento real)
train_data = pd.read_csv('train_set.csv')
test_data = pd.read_csv('test_set.csv')

# Use o LabelEncoder para mapear categorias de strings para IDs numéricos
label_encoder = LabelEncoder()

# Ajuste o LabelEncoder aos dados de treinamento e transforme as categorias
train_data['categories'] = label_encoder.fit_transform(train_data['categories'])
test_data['categories'] = label_encoder.transform(test_data['categories'])

# Hiperparâmetros
embedding_dim = 32
hidden_dim = 64
batch_size = 32
learning_rate = 0.001
num_epochs = 10

# Defina uma classe para o conjunto de dados
class MovieLensDataset(Dataset):
    def __init__(self, data):
        self.user_ids = torch.tensor(data['user'].values, dtype=torch.long, device=device)
        self.movie_ids = torch.tensor(data['movie'].values, dtype=torch.long, device=device)
        self.ratings = torch.tensor(data['rating'].values, dtype=torch.float, device=device)
        self.categories = torch.tensor(data['categories'].values, dtype=torch.long, device=device)

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        return {'user_id': self.user_ids[idx],
                'movie_id': self.movie_ids[idx],
                'rating': self.ratings[idx],
                'categories': self.categories[idx]}

# Crie conjuntos de treinamento e teste
train_dataset = MovieLensDataset(train_data)
test_dataset = MovieLensDataset(test_data)

# DataLoader personalizado para lidar com categorias variáveis
class CustomDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle)

    def collate_fn(self, batch):
        user_ids = torch.tensor([item['user_id'] for item in batch], dtype=torch.long, device=device)
        movie_ids = torch.tensor([item['movie_id'] for item in batch], dtype=torch.long, device=device)
        ratings = torch.tensor([item['rating'] for item in batch], dtype=torch.float, device=device)
        categories = torch.tensor([item['categories'] for item in batch], dtype=torch.long, device=device)

        return {'user_id': user_ids, 'movie_id': movie_ids, 'categories': categories, 'rating': ratings}

# Crie uma rede neural para recomendação de filmes
class RecommendationModel(nn.Module):
    def __init__(self, num_users, num_movies, num_categories, embedding_dim):
        super(RecommendationModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim).to(device)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim).to(device)
        self.category_embedding = nn.Embedding(num_categories, embedding_dim).to(device)
        self.fc1 = nn.Linear(embedding_dim * 3, hidden_dim).to(device)
        self.fc2 = nn.Linear(hidden_dim, 1).to(device)

    def forward(self, user_ids, movie_ids, categories):
        user_embeds = self.user_embedding(user_ids)
        movie_embeds = self.movie_embedding(movie_ids)
        category_embeds = self.category_embedding(categories)

        x = torch.cat([user_embeds, movie_embeds, category_embeds], dim=1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Crie uma instância da rede neural
model = RecommendationModel(num_users=max(train_data['user']) + 1,
                            num_movies=max(train_data['movie']) + 1,
                            num_categories=max(train_data['categories']) + 1,
                            embedding_dim=embedding_dim).to(device)

# Defina a função de perda e o otimizador
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Treinamento
train_loader = CustomDataLoader(train_dataset, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        user_ids, movie_ids, categories, ratings = batch['user_id'], batch['movie_id'], batch['categories'], batch['rating']
        optimizer.zero_grad()
        outputs = model(user_ids, movie_ids, categories)
        loss = criterion(outputs, ratings.view(-1, 1))
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

# Avaliação no conjunto de teste
test_loader = CustomDataLoader(test_dataset, batch_size=batch_size, shuffle=False)
model.eval()
with torch.no_grad():
    total_loss = 0
    total_samples = 0
    for batch in test_loader:
        user_ids, movie_ids, categories, ratings = batch['user_id'], batch['movie_id'], batch['categories'], batch['rating']
        outputs = model(user_ids, movie_ids, categories)
        loss = criterion(outputs, ratings.view(-1, 1))
        total_loss += loss.item() * len(user_ids)
        total_samples += len(user_ids)
    test_rmse = np.sqrt(total_loss / total_samples)
    print(f'Test RMSE: {test_rmse}')





