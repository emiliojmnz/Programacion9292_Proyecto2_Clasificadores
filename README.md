# Programacion9292_Proyecto2_Clasificadores

El código que se realizó en este proyecto sirve para analizar datos médicos y predecir si un tumor es benigno o maligno. Para esto, utiliza cinco métodos diferentes de "clasificación", estos métodos son: Regresión Logística, k-NN, SVM, Árbol de Decisión y Random Forest.

Primero, el código importa varias "herramientas" (bibliotecas) que necesita para funcionar. Pandas es una biblioteca que es útil para la manipulación y el análisis de datos, y en este caso nos ayuda a organizar los datos como en una tabla. Scikit-learn proporciona los métodos de clasificación y funciones para preparar los datos y evaluar los resultados. Matplotlib ayuda a crear gráficos para visualizar la información.

Luego, se define una clase llamada "Clasificadores" que tendrá ciertos atributos (características) y ciertos métodos (acciones). En este caso, la clase "Clasificadores" tiene la tarea de cargar los datos del archivo "cancer.csv", prepararlos para el análisis, entrenar los cinco métodos de clasificación y evaluar qué tan bien funcionan.

Dentro de la clase "Clasificadores", definimos las tres funciones principales:

__init__: Esta función se ejecuta automáticamente cuando se crea un nuevo "objeto" Clasificadores. Su trabajo es cargar los datos del archivo "cancer.csv", separarlos en dos grupos (uno para entrenar los métodos y otro para probarlos) y crear una lista con los cinco métodos de clasificación que se van a usar.

entrenar_y_evaluar: Esta función entrena cada uno de los cinco métodos con los datos de entrenamiento y luego los evalúa con los datos de prueba. Para la evaluación, se utilizan tres medidas: la precisión (qué tan a menudo acierta el método), la matriz de confusión (que muestra los tipos de errores que comete el método) y las puntuaciones de validación cruzada (que ayudan a asegurar que los resultados sean confiables).

comparar_arboles_decision: Esta función se centra en el método de "Árbol de Decisión". Prueba diferentes configuraciones del árbol (cambiando su "profundidad") y evalúa cuál funciona mejor. Luego, crea un gráfico para mostrar cómo cambia la precisión del método según la profundidad del árbol y selecciona el mejor árbol.

Los METODOS DE CLASIFICACION que hemos implementado en este código son los siguientes

1. Regresión Logística: Este método busca la mejor línea que separa los datos en dos grupos (benigno y maligno), la Regresión Logística traza una línea para que los puntos de un grupo queden de un lado y los del otro grupo del otro lado.

2. k-NN (k-Nearest Neighbors): Este método se basa en la idea de que los datos similares suelen estar cerca unos de otros. Para clasificar un nuevo dato, k-NN busca los "k" datos más cercanos y le asigna la clase más común entre ellos.

3. SVM (Máquinas de Soporte Vectorial): Este método busca la mejor superficie que separa los datos en dos grupos. A diferencia de la Regresión Logística, que busca una línea, SVM puede encontrar superficies más complejas, como planos o hiperplanos.
  
4. Árbol de Decisión: Este método crea un "árbol" de decisiones para clasificar los datos. Cada nodo del árbol representa una pregunta sobre los datos, y las ramas representan las posibles respuestas. Siguiendo las ramas del árbol, se llega a una decisión sobre la clase del dato.

5. Random Forest: Este método crea varios árboles de decisión y los combina para obtener una predicción más robusta.


Finalmente, la parte del código que dice if __name__ == "__main__": es la que se ejecuta cuando se inicia el programa. Aquí se crea un "objeto" Clasificadores, se entrenan y evalúan los métodos de clasificación y se compara el método de "Árbol de Decisión" para encontrar la mejor configuración. 
