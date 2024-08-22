# Importando as bibliotecas necessárias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn import tree
import matplotlib.pyplot as plt

# Carregar o arquivo CSV
df = pd.read_csv('../diabetes_prediction_dataset.csv')
df = df.drop(['gender', 'smoking_history'], axis=1)

# Visualizar as primeiras linhas do dataset
print(df.head())

# Supondo que o arquivo CSV tenha uma coluna 'target' que é a variável que queremos prever
# As features (variáveis independentes) são todas as outras colunas
X = df.drop('diabetes', axis=1)
y = df['diabetes']

# Dividir o conjunto de dados em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Criando o modelo de Árvore de Decisão
# Você pode ajustar o parâmetro 'max_depth' para limitar a profundidade da árvore
dtree = DecisionTreeClassifier(random_state=42)

# Treinando o modelo
dtree.fit(X_train, y_train)

# Fazendo previsões no conjunto de teste
y_pred = dtree.predict(X_test)

# Avaliando o modelo
print("Acurácia:", accuracy_score(y_test, y_pred))
print("Relatório de Classificação:\n", classification_report(y_test, y_pred))

# Visualizando a Árvore de Decisão
plt.figure(figsize=(20,10))
tree.plot_tree(dtree, filled=True, feature_names=X.columns, class_names=True, rounded=True)
plt.savefig("grafico-arvore-decisao.png")
#plt.show