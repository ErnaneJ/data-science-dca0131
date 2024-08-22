# Importando as bibliotecas necessárias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

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

# Normalizando as features (opcional, mas recomendado para KNN)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Criando o modelo KNN
knn = KNeighborsClassifier(n_neighbors=5)  # n_neighbors é o número de vizinhos

# Treinando o modelo
knn.fit(X_train, y_train)

# Fazendo previsões no conjunto de teste
y_pred = knn.predict(X_test)

# Avaliando o modelo
print("Acurácia:", accuracy_score(y_test, y_pred))
print("Relatório de Classificação:\n", classification_report(y_test, y_pred))
#plt.show()