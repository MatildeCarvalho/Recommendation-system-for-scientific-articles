import pandas as pd

train_data = pd.read_csv('Train_teste_split/train_set.csv')
test_data = pd.read_csv('Train_teste_split/test_set.csv')

# Crie conjuntos de pares (user, artigo) para treinamento e teste
train_pairs = set(zip(train_data['user'], train_data['article']))
test_pairs = set(zip(test_data['user'], test_data['article']))

# Verifique a sobreposição entre os pares
overlapping_pairs = train_pairs.intersection(test_pairs)

if len(overlapping_pairs) == 0:
    print("Não há a mesma combinação de user e artigo entre os conjuntos de treinamento e teste.")
else:
    print(f"Há {len(overlapping_pairs)} combinações de user e artigo em comum entre os conjuntos de treinamento e teste.")
    print("Estas são as combinações em comum:", overlapping_pairs)
