# Grant Nathaniel Keegan | A01700753 | logreg_base.py
# Aplica un modelo de regresión logística MEJORADO (sin frameworks) para predecir
# si futuros pacientes sufren de diabetes o no.

# Instalamos dependencias.
import pandas as pd # Para trabajar y manipular los archivos csv.
import numpy as np # Para las fórmulas matemáticas de los algoritmos. Ejemplo: np.dot: producto cruzado.
import math # Para operaciones como exp, log y sqrt.
import matplotlib.pyplot as plt # Para plotear los resultados en tablas.

# Leer el archivo modificado.
df = pd.read_csv("diabetes_data_modified.csv")
print(df.head())