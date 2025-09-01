# Grant Nathaniel Keegan | A01700753 | logreg_base.py
# Aplica un modelo de regresión logística (sin frameworks) para predecir
# si futuros pacientes sufren de diabetes o no.

# Instalamos dependencias.
import pandas as pd # Para trabajar y manipular los archivos csv.
import numpy as np # Para las fórmulas matemáticas de los algoritmos. Ejemplo: np.dot: producto cruzado.
import math # Para operaciones como exp, log y sqrt.
import matplotlib.pyplot as plt # Para plotear los resultados en tablas.

# Leer el archivo modificado.
df = pd.read_csv("diabetes_data_modified.csv")

# ================================================ #
# ===== DEFINICIÓN DE VARIABLES Y PARÁMETROS ===== #
# ================================================ #

# Asignamos las columnas de la tabla a X y Y. También definimos los pesos del algoritmo (params).
X = df['Age', 'Gender',
        'Polyuria', 'Polydipsia', 'sudden weight loss', 'weakness', 
        'Polyphagia', 'Genital thrush', 'visual blurring', 
        'Itching', 'Irritability', 'delayed healing', 
        'partial paresis', 'muscle stiffness', 'Alopecia', 'Obesity',]
Y = df["class"]
params = [0,0,0,0,0, # 15, para los 14 síntomas, más el peso de desplazamiento.
          0,0,0,0,0,
          0,0,0,0,0]

a = 0.1 # alpha. learning rate para entrenamiento. Modificar si es necesario.
epochs = 500 # epochs para entrenamiento. Modificar si es necesario.

# Definimos nuestras variables de entrenamiento (80% de los datos) y test (20%).
# Esto ayuda al modelo a predecir datos nuevos y evitar overfitting.
X_train = X

# ================================================ #
# ====== FUNCIONES PARA REGRESIÓN LOGÍSTICA ====== #
# ================================================ #

# Sigmoide e Hipótesis: predice la probabilidad (0 a 1) de que el paciente tiene diabetes.
def sigmoide(X):
    return 1 / (1 + np.exp(-X))

def hipotesis(params, sample):
    return sigmoide(-np.dot(params, sample))

# Función de costo / cross-entropy. Qué tan mal o bien está prediciento el modelo. Usa hipótesis.
def costo(params, samples, Y):
    predicciones = []


# Gradiente Descendiente: Ajusta los pesos para que el modelo sea más acertado.
# R
#def gradiente_descendiente(X, Y, params, a, epochs):
 #   for i in range(len(params)):

def entrenamiento(params, X, Y, a , epochs):
    for epoch in range(epochs):
        for i in range(len(X)):
            prediccion = hipotesis(params, X[i])
            error = prediccion - Y[i]



# ================================================ #
# ============= CORREMOS EL CÓDIGO =============== #
# ================================================ #



# Corremos el código usando gradiente descendiente hasta que params 
#for epoch in epochs:
 #   temp_params = list(params)
  #  params = gradiente_descendiente(params, a, X, Y) # Ejecuta la función de gradiente descendiente.
    

# ================================================ #
# =========== ES HORA DE GRAFICAR! =============== #
# ================================================ #

# Vamos a hacer una gráfica de regresión logística para cada una de las 16 variables.

# Crear subplots
fig, axs = plt.subplots(4, 4, figsize=(20, 18))  # 4x4 = 16 gráficas
axs = axs.flatten()

# Recorrer cada columna
for i, columna in enumerate(X):
    x = df[columna]

    # Calcular regresión lineal
    m, b = np.polyfit(x, y, 1)  # m = pendiente, b = intercepto

    axs[i].scatter(x, y, alpha=0.3, color='orange', label='Pacientes')
    axs[i].plot(x, m * x + b, color='blue', label='Línea de regresión')

    axs[i].set_title(columna, fontsize=12)
    axs[i].set_xlabel(columna)
    axs[i].set_ylabel('Diabetes (0 = No, 1 = Sí)')
    axs[i].legend()
    axs[i].grid(True)

plt.tight_layout()
plt.suptitle("Regresión lineal por variable (1 gráfica por cada X)", fontsize=18, y=1.02)
plt.show()