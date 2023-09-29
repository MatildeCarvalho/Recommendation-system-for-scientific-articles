import pandas as pd
import numpy as np



def load_dataframe(file_path):
    return pd.read_csv(file_path)

articles_df = load_dataframe('file_final\\articles_combined.csv')
interactions_df = load_dataframe('file_final\\users_data.csv')

articles_df = articles_df.drop(['old_doc.id'], axis=1)
interactions_df = interactions_df.drop(['old_doc.id'], axis=1)

train_set = pd.DataFrame(columns=['user', 'article', 'rating', 'entities'])
test_set = pd.DataFrame(columns=['user', 'article', 'rating', 'entities'])



# Supondo que você tenha um dataframe chamado "interactions_df" com todas as interações dos users
users = interactions_df['user'].unique()

def create_article_sample(interactions_df, articles_df, user_id, num_samples):
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
        'rating': 0
    })

    # Adicionar as entidades correspondentes aos artigos no DataFrame de amostra
    sample_df['entities'] = sample_df['article'].map(articles_df.set_index('doc.id')['entities'])
    return sample_df


def select_test_rows(df, test_percentage):
    # Calcula o tamanho do conjunto de teste
    test_size = int(len(df) * test_percentage)

    # Embaralha o dataframe para seleção aleatória
    df_shuffled = df.sample(frac=1, random_state=2)

    # Divide o dataframe em conjunto de teste e treinamento
    test_rows = df_shuffled.head(test_size)
    train_rows = df_shuffled.tail(len(df_shuffled) - test_size)

    return train_rows, test_rows


for user in users:
    user_rows = interactions_df.loc[interactions_df['user'] == user]

    # Split the user rows into train and test sets per user
    train_rows, test_rows = select_test_rows(user_rows, test_percentage=0.2)
    # Create the sample for the user
    sample_df = create_article_sample(interactions_df, articles_df, user, num_samples=len(test_rows)) # 1/3 ou 1/2... antes tinha 2*len(test_rows) para fazer 2/3 de artigos que nao sabia

    # Append the sample to the test set
    test_rows_with_sample = pd.concat([test_rows, sample_df])

    train_set = pd.concat([train_set, train_rows])
    test_set = pd.concat([test_set, test_rows_with_sample])

train_set.to_csv('train_teste_split\\train_set.csv', index=False)
test_set.to_csv('train_teste_split\\test_set.csv', index=False)

