"""
Programación 9292
Proyecto 2 - Clasificadores
autor: Jiménez Malvaez Raúl Emilio
"""
import pandas as pd
from collections import Counter
import math

def distancia_euclidiana(x1, x2):
  """Calcula la distancia euclidiana entre dos puntos."""
  distancia = 0
  for i in range(len(x1)):
    distancia += (x1[i] - x2[i]) ** 2
  return math.sqrt(distancia)

def knn(X_train, y_train, X_test, k):
  """Implementa el algoritmo k-NN."""
  y_pred = []
  for punto_test in X_test:
    distancias = [distancia_euclidiana(punto_test, punto_train) for punto_train in X_train]
    k_vecinos_indices = sorted(range(len(distancias)), key=lambda i: distancias[i])[:k]
    k_vecinos_etiquetas = [y_train[i] for i in k_vecinos_indices]
    etiqueta_predicha = Counter(k_vecinos_etiquetas).most_common(1)[0][0]
    y_pred.append(etiqueta_predicha)
  return y_pred

def accuracy_score(y_true, y_pred):
  """Calcula la precisión."""
  correctos = 0
  for i in range(len(y_true)):
    if y_true[i] == y_pred[i]:
      correctos += 1
  return correctos / len(y_true)

def confusion_matrix(y_true, y_pred):
  """Calcula la matriz de confusión."""
  clases = set(y_true)
  matriz = [[0 for _ in clases] for _ in clases]
  for i in range(len(y_true)):
    verdadera = y_true[i]
    prediccion = y_pred[i]
    matriz[verdadera][prediccion] += 1
  return matriz

# Cargar los datos
data = pd.read_excel("C:\Users\emili\Downloads\cancer.csv")

# Separar las características (X) y la variable objetivo (y)
X = data.drop("diagnosis", axis=1)
y = data["diagnosis"]

# Codificar la variable objetivo a valores numéricos (0 para 'B' y 1 para 'M')
y = y.map({'B': 0, 'M': 1})

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convertir los datos a listas para facilitar el acceso a los elementos
X_train = X_train.values.tolist()
y_train = y_train.values.tolist()
X_test = X_test.values.tolist()
y_test = y_test.values.tolist()

# Predecir con k-NN (k=5 por ejemplo)
y_pred = knn(X_train, y_train, X_test, k=5)

# Calcular la precisión
precision = accuracy_score(y_test, y_pred)

# Calcular la matriz de confusión
matriz_confusion = confusion_matrix(y_test, y_pred)

# Mostrar los resultados
print(f"Clasificador: k-NN")
print(f"Precisión: {precision}")
print(f"Matriz de Confusión:\n{matriz_confusion}\n")