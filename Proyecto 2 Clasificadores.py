"""
Programación 9292
Proyecto 2 - Clasificadores
autor: Jiménez Malvaez Raúl Emilio
"""
#Comenzamos por importar todas las bibliotecas necesarias
import pandas as pd #Para la manipulación de datos
from sklearn.model_selection import train_test_split, cross_val_score #Para dividir los datos y realizar validación cruzada
from sklearn.linear_model import LogisticRegression #Modelo de Regresión Logística
from sklearn.neighbors import KNeighborsClassifier #Modelo de k-Nearest Neighbors
from sklearn.svm import SVC #Modelo de Máquinas de Soporte Vectorial
from sklearn.tree import DecisionTreeClassifier, plot_tree #Modelo de Árbol de Decisión y función para visualizarlo
from sklearn.ensemble import RandomForestClassifier #Modelo de Random Forest
from sklearn.metrics import accuracy_score, confusion_matrix #Métricas para evaluar el rendimiento
import matplotlib.pyplot as plt #Para la visualización de datos a través de gráficas


class Clasificadores:
    """
    Clase para entrenar, evaluar y comparar diferentes clasificadores
    utilizando la base de datos "cancer.csv".
    """

    def __init__(self, data_path="cancer.csv"):
        """
        Inicializamos la clase Clasificadores.

        Argumento:
        data_path: Ruta al archivo de datos (csv).
        Como el archivo del código y de la base de datos .csv se hallarán en la misma carpeta/repositorio, basta con "cancer.csv"
        """
        try:
           #Cargamos los datos del archivo CSV usando pandas
            self.data = pd.read_csv(data_path)

           #Separamos las características (X) de la variable objetivo (y)
            self.X = self.data.drop("diagnosis", axis=1) #Eliminamos la columna "diagnosis" del DataFrame para obtener las características
            self.y = self.data["diagnosis"].map({'B': 0, 'M': 1}) #Mapeamos la columna "diagnosis": 'B' a 0 (benigno) y 'M' a 1 (maligno)

           #Dividimos los datos en conjuntos de entrenamiento y prueba
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=0.2, random_state=42 #80% para entrenamiento, 20% para prueba, semilla aleatoria para reproducibilidad
            )

           #Definimos los clasificadores a utilizar
            self.clasificadores = {
                "Regresión Logística": LogisticRegression(),
                "k-NN": KNeighborsClassifier(),
                "SVM": SVC(),
                "Árbol de Decisión": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
            }

        except FileNotFoundError: #Manejamos la excepción si no se encuentra el archivo
            print(f"Error: No se pudo encontrar el archivo '{data_path}'")
            raise #Volvemos a lanzar la excepción para detener la ejecución
        except Exception as e: #Manejamos cualquier otra excepción
            print(f"Error al cargar o procesar los datos: {e}")
            raise #Volvemos a lanzar la excepción para detener la ejecución

   #Entrenamos y evaluamos los clasificadores.
    def entrenar_y_evaluar(self):
        resultados = {} #Creamos un diccionario vacío para almacenar los resultados
        for nombre, clasificador in self.clasificadores.items(): #Iteramos sobre cada clasificador
            try:
               #Entrenamos el clasificador con los datos de entrenamiento
                clasificador.fit(self.X_train, self.y_train)

               #Realizamos predicciones con el clasificador usando los datos de prueba
                y_pred = clasificador.predict(self.X_test)

               #Evaluamos el rendimiento del clasificador
                precision = accuracy_score(self.y_test, y_pred) #Calculamos la precisión
                matriz_confusion = confusion_matrix(self.y_test, y_pred) #Calculamos la matriz de confusión
                cv_scores = cross_val_score(clasificador, self.X, self.y, cv=5) #Calculamos las puntuaciones de validación cruzada (5 folds)

               #Guardamos los resultados en el diccionario
                resultados[nombre] = {
                    "Precisión": precision,
                    "Matriz de Confusión": matriz_confusion,
                    "Cross-validation scores": cv_scores,
                }
            except Exception as e: #Manejamos cualquier excepción durante el entrenamiento o la evaluación
                print(f"Error al entrenar o evaluar {nombre}: {e}")
                resultados[nombre] = {"Error": str(e)} #Guardamos el mensaje de error en el diccionario
        return resultados #Devolvemos el diccionario con los resultados

    def comparar_arboles_decision(self):
        """
        Comparamos diferentes árboles de decisión con diferentes
        profundidades y seleccionamos el mejor basado en la precisión
        en el conjunto de prueba.
        Esta función nos regresará el mejor árbol de decisión encontrado.
        """
        profundidades = range(1, 11) #Creamos una lista de profundidades del 1 al 11
        resultados_arboles = [] #Creamos una lista para almacenar los resultados

       #Iteramos sobre cada profundidad
        for profundidad in profundidades:
           #Creamos un árbol de decisión con la profundidad actual
            arbol = DecisionTreeClassifier(max_depth=profundidad, random_state=42) #Semilla aleatoria para reproducibilidad

           #Entrenamos el árbol con los datos de entrenamiento
            arbol.fit(self.X_train, self.y_train)

           #Realizamos predicciones con el árbol usando los datos de prueba
            y_pred = arbol.predict(self.X_test)

           #Calculamos la precisión del árbol y la guardamos en la lista
            precision = accuracy_score(self.y_test, y_pred)
            resultados_arboles.append(precision)

       #Creamos una gráfica para comparar la precisión de los árboles con diferentes profundidades
        plt.figure(figsize=(10, 6))
        plt.plot(profundidades, resultados_arboles, marker='o') #Creamos un gráfico de línea con marcadores
        plt.xlabel("Profundidad del árbol") #Etiqueta del eje x
        plt.ylabel("Precisión") #Etiqueta del eje y
        plt.title("Comparación de árboles de decisión con diferentes profundidades") #Definimos el título del gráfico
        plt.grid(True) #Agregamos una cuadrícula al gráfico
        plt.show() #Mostramos el gráfico

       #Seleccionamos el mejor árbol de decisión (el que tiene la mayor precisión)
        mejor_profundidad = profundidades[resultados_arboles.index(max(resultados_arboles))] #Encontramos la profundidad con la mayor precisión
        mejor_arbol = DecisionTreeClassifier(max_depth=mejor_profundidad, random_state=42) #Creamos un nuevo árbol con la mejor profundidad
        mejor_arbol.fit(self.X_train, self.y_train) #Entrenamos el mejor árbol

        print(f"Mejor profundidad del árbol: {mejor_profundidad}") #Imprimimos la mejor profundidad

       #Visualizamos el mejor árbol de decisión
        plt.figure(figsize=(12, 8))
        plot_tree(mejor_arbol, feature_names=self.X.columns, class_names=["B", "M"], filled=True) #Creamos una visualización del árbol
        plt.title("Árbol de decisión óptimo") #Definimos el título del gráfico
        plt.show() #Mostramos el gráfico

        return mejor_arbol #Devolvemos el mejor árbol de decisión


#Esta es la parte más importante del código, pues ejecuta todo lo que definimos anteriormente dentro de un __main__
if __name__ == "__main__":
    try:
       #Creamos una instancia de la clase Clasificadores
        clasificadores = Clasificadores()

       #Entrenamos y evaluamos los clasificadores
        resultados = clasificadores.entrenar_y_evaluar()

       #Imprimimos los resultados de cada clasificador
        for nombre, resultado in resultados.items(): #Iteramos sobre los resultados
            print(f"\nClasificador: {nombre}") #Imprimimos el nombre del clasificador
            for metrica, valor in resultado.items(): #Iteramos sobre las métricas
                print(f"{metrica}: {valor}") #Imprimimos el nombre de la métrica y su valor

       #Comparamos los árboles de decisión y encontramos el mejor
        mejor_arbol = clasificadores.comparar_arboles_decision()
        print(f"\nMejor árbol de decisión: {mejor_arbol}") #Imprimimos el mejor árbol

    except Exception as e: #Manejamos cualquier excepción en el bloque principal
        print(f"Error en el programa principal: {e}") #Imprimimos el mensaje de error
