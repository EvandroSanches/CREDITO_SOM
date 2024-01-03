from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler
from pandasgui import show
import pandas as pd
import matplotlib.pylab as plb
import numpy as np

#Carregando e tratando base de dados
base = pd.read_csv('credit_data.csv')
base = base.dropna()
base.age[base.age < 0] = base.age[base.age < 0] * -1
classe = base.default
dados = base.drop('default', axis=1)

#Normalizando dados
normalizador_dados = MinMaxScaler()
normalizador_classe = MinMaxScaler()

dados = normalizador_dados.fit_transform(dados)
classe = np.expand_dims(classe, axis=1)
classe = normalizador_classe.fit_transform(classe)

#Criando modelo
som = MiniSom(x=15, y=15, input_len=4, learning_rate=0.5, sigma=1, random_seed=8)

#Treinando modelo
som.random_weights_init(dados)
som.train_random(data=dados, num_iteration=1000)

#Plotando mapa do modelo
plb.pcolor(som.distance_map().T)
plb.colorbar()

#Definindo marcadores
markers = ['s', 'o']
colors = ['b', 'r']

#Inserindo marcadores no mapa
for i, x in enumerate(dados):
    w = som.winner(x)

    plb.plot(w[0] + 0.5, w[1] + 0.5, markers[int(classe[i])], markerfacecolor='None',
             markersize=10, markeredgecolor=colors[int(classe[i])], markeredgewidth=2)

plb.show()

#Definindo valores fora do padrão
mapeanmento = som.win_map(dados)
outliers = np.concatenate((mapeanmento[(10,1)],mapeanmento[(10,2)]))
outliers = normalizador_dados.inverse_transform(outliers)


credito = []

#Ajustando valores
for i in range(len(base)):
    for j in range(len(outliers)):
        if base.iloc[i,0] == int(round(outliers[j,0])):
            credito.append(base.iloc[i,4])

credito = np.asarray(credito)

#Defenindo registros de maus devedores
suspeitos = np.column_stack((outliers, credito))
df_suspeitos = pd.DataFrame(suspeitos)


show(df_suspeitos)
