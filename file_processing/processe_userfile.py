import pandas as pd

# Lê o arquivo JSON 'users.json'
file_path = 'file_creation\\users.json'
df = pd.read_json(file_path)

# Use o método drop para remover as linhas onde 'article' é igual a 12479
df = df[df['article'] != 12479]

# Lê o arquivo CSV 'articles_combined.csv'
articles_df = pd.read_csv('file_processing\\articles_combined.csv')

# Adicione uma coluna 'old_doc.id' em 'articles_df' para manter os valores antigos da coluna 'doc.id'
articles_df['old_doc.id'] = articles_df['doc.id']

# Use o método merge para juntar os dois dataframes
# Realize o merge com base nas colunas 'article' e 'doc.id'
merged_df = pd.merge(df, articles_df, left_on='article', right_on='doc.id', how='left')

# Lista das colunas que você deseja remover
colunas_para_remover = ['doc.id', 'title', 'citeulike.id', 'raw.title', 'raw.abstract', 'clean_abstract']

# Use o método drop para remover as colunas especificadas
merged_df = merged_df.drop(columns=colunas_para_remover)

# Atualize a coluna 'doc.id' em 'articles_df' para começar com 0
articles_df['doc.id'] = range(0, len(articles_df))

# Salva o DataFrame 'articles_df' de volta no arquivo CSV original
articles_df.to_csv('file_final\\articles_combined.csv', index=False)

# Agora, atualize o DataFrame 'merged_df' com a nova numeração
# Primeiro, crie um mapeamento entre os valores antigos e os novos na coluna 'article'
article_mapping = {old_id: new_id for old_id, new_id in zip(articles_df['old_doc.id'], range(0, len(articles_df)))}
print(article_mapping)
# Atualize os valores na coluna 'article' do DataFrame 'merged_df' usando o mapeamento
merged_df['article'] = merged_df['article'].map(article_mapping)

# Trate valores não finitos na coluna 'article'
# Use o método fillna para substituir os valores nulos (que não têm mapeamento) por 0
merged_df['article'] = merged_df['article'].fillna(0).astype(int)

# Verifique o resultado
print(merged_df.head())

# Verifique o resultado
print(merged_df.head())

total_unique_doc_ids = merged_df['article'].nunique()
print(f"Total de articles diferentes: {total_unique_doc_ids}")

total_unique_users_ids = merged_df['user'].nunique()
print(f"Total de users diferentes: {total_unique_users_ids}")

# Salve o DataFrame 'merged_df' com as alterações no arquivo CSV final
merged_df.to_csv('file_final\\users_data.csv', index=False)



