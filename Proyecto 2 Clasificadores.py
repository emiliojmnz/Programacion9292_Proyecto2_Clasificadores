"""
Programación 9292
Proyecto 2 - Clasificadores
autor: Jiménez Malvaez Raúl Emilio
"""
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt


class Clasificadores:
    """
    Clase para entrenar, evaluar y comparar diferentes clasificadores 
    utilizando la base de datos "cancer.xlsx".
    """

    def __init__(self, data_path="cancer.xlsx"):
        """
        Inicializa la clase Clasificadores.

        Args:
          data_path (str): Ruta al archivo de datos (xlsx). Por defecto 
                            es "cancer.xlsx".
        """
        try:
            self.data = pd.read_excel(data_path, engine='openpyxl')
            self.X = self.data.drop("diagnosis", axis=1)
            self.y = self.data["diagnosis"].map({'B': 0, 'M': 1})
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=0.2, random_state=42
            )
            self.clasificadores = {
                "Regresión Logística": LogisticRegression(),
                "k-NN": KNeighborsClassifier(),
                "SVM": SVC(),
                "Árbol de Decisión": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
            }
        except FileNotFoundError:
            print(f"Error: No se pudo encontrar el archivo '{data_path}'")
            raise
        except Exception as e:
            print(f"Error al cargar o procesar los datos: {e}")
            raise

    def entrenar_y_evaluar(self):
        """
        Entrena y evalúa los clasificadores.

        Returns:
          dict: Diccionario con los resultados de cada clasificador, 
                incluyendo precisión, matriz de confusión y 
                puntuaciones de validación cruzada.
        """
        resultados = {}
        for nombre, clasificador in self.clasificadores.items():
            try:
                # Entrenar
                clasificador.fit(self.X_train, self.y_train)
                # Predecir
                y_pred = clasificador.predict(self.X_test)
                # Evaluar
                precision = accuracy_score(self.y_test, y_pred)
                matriz_confusion = confusion_matrix(self.y_test, y_pred)
                cv_scores = cross_val_score(clasificador, self.X, self.y, cv=5)
                resultados[nombre] = {
                    "Precisión": precision,
                    "Matriz de Confusión": matriz_confusion,
                    "Cross-validation scores": cv_scores,
                }
            except Exception as e:
                print(f"Error al entrenar o evaluar {nombre}: {e}")
                resultados[nombre] = {"Error": str(e)}
        return resultados

    def comparar_arboles_decision(self):
        """
        Compara diferentes árboles de decisión con diferentes 
        profundidades y selecciona el mejor basado en la precisión 
        en el conjunto de prueba.

        Returns:
          DecisionTreeClassifier: El mejor árbol de decisión encontrado.
        """
        profundidades = range(1, 11)
        resultados_arboles = []
        for profundidad in profundidades:
            arbol = DecisionTreeClassifier(max_depth=profundidad, random_state=42)
            arbol.fit(self.X_train, self.y_train)
            y_pred = arbol.predict(self.X_test)
            precision = accuracy_score(self.y_test, y_pred)
            resultados_arboles.append(precision)

        # Gráfica de comparación
        plt.figure(figsize=(10, 6))
        plt.plot(profundidades, resultados_arboles, marker='o')
        plt.xlabel("Profundidad del árbol")
        plt.ylabel("Precisión")
        plt.title("Comparación de árboles de decisión con diferentes profundidades")
        plt.grid(True)
        plt.show()

        # Seleccionar el mejor árbol (ejemplo: con mayor precisión)
        mejor_profundidad = profundidades[resultados_arboles.index(max(resultados_arboles))]
        mejor_arbol = DecisionTreeClassifier(max_depth=mejor_profundidad, random_state=42)
        mejor_arbol.fit(self.X_train, self.y_train)

        print(f"Mejor profundidad del árbol: {mejor_profundidad}")

        # Visualizar el mejor árbol
        plt.figure(figsize=(12, 8))
        plot_tree(mejor_arbol, feature_names=self.X.columns, class_names=["B", "M"], filled=True)
        plt.title("Árbol de decisión óptimo")
        plt.show()

        return mejor_arbol


if __name__ == "__main__":
    try:
        clasificadores = Clasificadores("cancer.xlsx")
        resultados = clasificadores.entrenar_y_evaluar()

        # Imprimir los resultados de cada clasificador
        for nombre, resultado in resultados.items():
            print(f"\nClasificador: {nombre}")
            for metrica, valor in resultado.items():
                print(f"{metrica}: {valor}")

        mejor_arbol = clasificadores.comparar_arboles_decision()
        print(f"\nMejor árbol de decisión: {mejor_arbol}")

    except Exception as e:
        print(f"Error en el programa principal: {e}")
