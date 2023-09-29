import pandas as pd
import matplotlib.pyplot as plt

# Ler os dados do arquivo CSV
data = pd.read_csv('resultado_final.csv')

# Extrair os valores de K e as métricas
k_values = data['K']
precision_tf_idf = data['Precision (TF-IDF)']
precision_entity = data['Precision (Entity)']
recall_tf_idf = data['Recall (TF-IDF)']
recall_entity = data['Recall (Entity)']

# Criar o gráfico de Precision
plt.figure(figsize=(8, 6))
plt.plot(k_values, precision_tf_idf, marker='o', label='TF-IDF')
plt.plot(k_values, precision_entity, marker='o', label='Entity')
plt.xlabel('K')
plt.ylabel('Precision')
plt.title('Precision vs. K')
plt.xticks(range(1, 6))  # Personalizar o eixo X para mostrar valores inteiros de 1 a 5
plt.legend()
plt.grid(False)  # Desabilitar a grade
plt.tight_layout()  # Ajustar o layout

# Salvar o gráfico de Precision em um arquivo
plt.savefig('precision_plot.png')

# Criar o gráfico de Recall
plt.figure(figsize=(8, 6))
plt.plot(k_values, recall_tf_idf, marker='o', label='TF-IDF')
plt.plot(k_values, recall_entity, marker='o', label='Entity')
plt.xlabel('K')
plt.ylabel('Recall')
plt.title('Recall vs. K')
plt.xticks(range(1, 6))  # Personalizar o eixo X para mostrar valores inteiros de 1 a 5
plt.legend()
plt.grid(False)  # Desabilitar a grade
plt.tight_layout()  # Ajustar o layout

# Salvar o gráfico de Recall em um arquivo
plt.savefig('recall_plot.png')

# Mostrar os gráficos
plt.show()
