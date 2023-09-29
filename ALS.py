# pip install lenskit
# drive implicit 
from lenskit.algorithms import als
import pandas as pd
from lenskit import batch, topn, util
from lenskit import crossfold as xf
from lenskit.algorithms import als, item_knn
import numpy as np
import pandas as pd
import numpy as np
from lenskit import batch, topn, util
from lenskit import crossfold as xf
from lenskit.algorithms import als, Recommender, item_knn as knn
from lenskit.metrics import topn as tnmetrics
import pandas as pd
import numpy as np
from lenskit import batch, topn, util
from lenskit import crossfold as xf
from lenskit.algorithms import als, Recommender, item_knn as knn
from lenskit.metrics import topn as tnmetrics

# definir a faixa de valores a serem testados para cada parâmetro
alphas = [10, 50, 100]
factors = [10, 20, 30]
regularizations = [0.1, 0.01, 0.001]

# inicializar uma lista para armazenar os resultados
results = []

# iterar sobre todas as combinações de parâmetros
for alpha in alphas:
    for factor in factors:
        for regularization in regularizations:
            # criar o algoritmo ALS com os parâmetros atuais
            algo_als = als.ImplicitMF(features=factor, reg=regularization, weight=alpha)
            # avaliar o desempenho do algoritmo ALS usando a validação cruzada
            all_recs = []
            test_data = []
            for train, test in xf.partition_users(ratings[['user', 'item', 'rating']], 3, xf.SampleFrac(0.2)):
                test_data.append(test)
                all_recs.append(eval('ALS', algo_als, train, test))

            all_recs = pd.concat(all_recs, ignore_index=True)
            test_data = pd.concat(test_data, ignore_index=True)

            # calcular as métricas de desempenho
            rla = topn.RecListAnalysis()
            rla.add_metric(tnmetrics.ndcg)
            results_df = rla.compute(all_recs, test_data).agg({'ndcg': 'mean'})

            # armazena os resultados e os parâmetros atuais
            params = {'alpha': alpha, 'factors': factor, 'regularization': regularization}
            results_df = pd.concat([pd.Series(params), results_df])
            results.append(results_df)

# encontra a combinação de parâmetros com o melhor desempenho
results_df = pd.concat(results, axis=1).T
best_params = results_df.iloc[results_df['ndcg'].idxmax(), :]

print('Melhores parâmetros: ')
print(best_params)

#################################
# Melhores parâmetros: 
# alpha             100.000000
# factors            30.000000
# regularization      0.010000
# ndcg                0.191522
# Name: 25, dtype: float64
################################
import pandas as pd
import numpy as np
from lenskit import batch, topn, util
from lenskit import crossfold as xf
from lenskit.algorithms import als, Recommender, item_knn as knn
from lenskit.metrics import topn as tnmetrics




data = pd.read_json('/content/drive/MyDrive/COOL/OFICIAL/FINAL/FILES/final_uartel.json')
data = data.drop(['title','entities'], axis=1)
ratings = data.reindex(np.random.permutation(data.index)).reset_index(drop=True)
ratings = ratings.rename(columns={"article": "item"})



# criar um algoritmo de recomendação ALS com 50 fatores latentes

algo_als = als.ImplicitMF(features=30, reg=0.01, weight=100)


def eval(aname, algo, train, test):
    fittable = util.clone(algo)
    fittable = Recommender.adapt(fittable)
    fittable.fit(train)
    users = test.user.unique()
    # now we run the recommender
    recs = batch.recommend(fittable, users, 100)
    # add the algorithm name for analyzability
    recs['Algorithm'] = aname
    return recs


all_recs = []
test_data = []
# avaliar o desempenho do algoritmo ALS usando a validação cruzada
results = []
for train, test in xf.partition_users(ratings[['user', 'item', 'rating']], 5, xf.SampleFrac(0.2)):
    test_data.append(test)
    all_recs.append(eval('ALS', algo_als, train, test))




all_recs = pd.concat(all_recs, ignore_index=True)
test_data = pd.concat(test_data, ignore_index=True)


rla = topn.RecListAnalysis()
rla.add_metric(topn.ndcg, name='ndcg_1', k=1)
rla.add_metric(topn.ndcg, name='ndcg_2', k=2)
rla.add_metric(topn.ndcg, name='ndcg_3', k=3)
rla.add_metric(topn.ndcg, name='ndcg_4', k=4)
rla.add_metric(topn.ndcg, name='ndcg_5', k=5)
rla.add_metric(topn.ndcg, name='ndcg_6', k=6)
rla.add_metric(topn.ndcg, name='ndcg_7', k=7)
rla.add_metric(topn.ndcg, name='ndcg_8', k=8)
rla.add_metric(topn.ndcg, name='ndcg_9', k=9)
rla.add_metric(topn.ndcg, name='ndcg_10', k=10)
rla.add_metric(topn.precision, name='precision_1', k=1)
rla.add_metric(topn.precision, name='precision_2', k=2)
rla.add_metric(topn.precision, name='precision_3', k=3)
rla.add_metric(topn.precision, name='precision_4', k=4)
rla.add_metric(topn.precision, name='precision_5', k=5)
rla.add_metric(topn.precision, name='precision_6', k=6)
rla.add_metric(topn.precision, name='precision_7', k=7)
rla.add_metric(topn.precision, name='precision_8', k=8)
rla.add_metric(topn.precision, name='precision_9', k=9)
rla.add_metric(topn.precision, name='precision_10', k=10)
rla.add_metric(topn.recall, name='recall_1', k=1)
rla.add_metric(topn.recall, name='recall_2', k=2)
rla.add_metric(topn.recall, name='recall_3', k=3)
rla.add_metric(topn.recall, name='recall_4', k=4)
rla.add_metric(topn.recall, name='recall_5', k=5)
rla.add_metric(topn.recall, name='recall_6', k=6)
rla.add_metric(topn.recall, name='recall_7', k=7)
rla.add_metric(topn.recall, name='recall_8', k=8)
rla.add_metric(topn.recall, name='recall_9', k=9)
rla.add_metric(topn.recall, name='recall_10', k=10)
results = rla.compute(all_recs, test_data)
results.head()
results.groupby('Algorithm').mean()

import matplotlib.pyplot as plt

# extrair as métricas ndcg_1 até ndcg_10 da tabela results
ndcg_values = results.filter(regex='^ndcg_\d+', axis=1).mean()

# extrair as métricas precision_1 até precision_10 da tabela results
precision_values = results.filter(regex='^precision_\d+', axis=1).mean()

# extrair as métricas recall_1 até recall_10 da tabela results
recall_values = results.filter(regex='^recall_\d+', axis=1).mean()

# valores de k para o eixo x
k_values = range(1, 11)

# plotar as métricas em um único gráfico
plt.plot(k_values, ndcg_values, label='NDCG')
plt.plot(k_values, precision_values, label='Precision')
plt.plot(k_values, recall_values, label='Recall')
plt.legend()
plt.xlabel('k')
plt.ylabel('Valor da métrica')
plt.title('Desempenho do algoritmo para diferentes valores de k')
plt.show()