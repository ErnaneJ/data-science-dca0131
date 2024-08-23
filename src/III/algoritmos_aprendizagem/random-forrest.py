# Importando as bibliotecas necessárias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criando o modelo de Floresta Aleatória
# Você pode ajustar o parâmetro 'n_estimators' para definir o número de árvores na floresta
rfc = RandomForestClassifier(n_estimators=100, random_state=42)

# Treinando o modelo
rfc.fit(X_train, y_train)

# Fazendo previsões no conjunto de teste
y_pred = rfc.predict(X_test)

# Avaliando o modelo
print("Acurácia:", accuracy_score(y_test, y_pred))
print("Relatório de Classificação:\n", classification_report(y_test, y_pred))

# Importância das features
importances = rfc.feature_importances_
indices = importances.argsort()[::-1]

# Plotando a importância das features
plt.figure(figsize=(12,6))
sns.barplot(x=importances[indices], y=X.columns[indices])
plt.title('Importância das Features')
plt.savefig("grafico-ramdom-forrest.png")
plt.show()