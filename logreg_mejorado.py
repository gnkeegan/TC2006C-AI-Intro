# =================================================== #
# Logistic Regression desde cero (binario) (MEJORADO) #
# ============== Grant Nathaniel Keegan ============= #
# =================================================== #

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ================================================ #
# ===== DEFINICIÓN DE VARIABLES Y PARÁMETROS ===== #
# ================================================ #

# 1. Cargar dataset
df = pd.read_csv("diabetes_data_modified.csv")

# Variables de entrada (X) y salida (Y)
X = df[["Age", "Gender", 
        "Polyuria", "Polydipsia", "sudden weight loss", "weakness", "Polyphagia",
        "Genital thrush", "visual blurring", "Itching", "Irritability",
        "delayed healing", "partial paresis", "muscle stiffness", "Alopecia", "Obesity"
]]
Y = df["class"]

# ================================================ #
# ======== PREPARACIóN DE DATOS CON NUMPY ======== #
# = DIVISION DE DATOS EN TRAIN, TEST, VALIDATION = #
# ================================================ #

# Normalizamos X para que todas las variables estén en la misma escala.
X = (X - X.mean()) / X.std()

# Convertir a numpy.
X = np.array(X); Y = np.array(Y)

# Agregar bias (columna de 1s) a los datos.
X = np.c_[np.ones(X.shape[0]), X]

# 2. Dividir datos en train (60%), val (20%), test (20%) usando np.split.
train_split = int(0.75 * len(X))
val_split = int(0.15 * len(X))
test_split = int(0.15 * len(X))

X_train, X_val, X_test = np.split(X, [train_split, train_split + val_split])
Y_train, Y_val, Y_test = np.split(Y, [train_split, train_split + val_split])

# ================================================ #
# = CONSTRUCCIóN DEL ALGORITMO A BASE DE FUNCIONES #
# ================================================ #

# Función de sigmoide para hipótesis.
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Función de hipótesis.
def h(params, samples):
    return sigmoid(np.dot(samples, params))

# Función de costo (entropía cruzada).
def cost_function(params, samples, y, lambda_reg):
    predictions = h(params, samples)
    m = len(samples)
    cost = -np.mean(y * np.log(predictions + 1e-9) + (1 - y) * np.log(1 - predictions + 1e-9))
    reg_term = (lambda_reg / (2 * m)) * np.sum(params[1:]**2)  # no penaliza bias
    return cost + reg_term

# Función de gradiente descendiente.
def GD(params, samples, y, a, lambda_reg):
    m = len(samples)
    predictions = h(params, samples)
    errors = predictions - y
    gradient = np.dot(samples.T, errors) / m
    gradient[1:] += (lambda_reg / m) * params[1:]  # no regulariza bias
    params -= a * gradient
    return params

# Función de predicción. Regresa un resultado de 1 o 0.
def predict(params, samples):
    return (h(params, samples) >= 0.5).astype(int) # Regresa 1 o 0 basado en el umbral de decisión.

# Función para calcular accuracy (gráfica 4).
def accuracy(params, samples, labels):
    predictions = predict(params, samples)
    return np.mean(predictions == labels)

# ================================================== #
# = ENTRENAMIENTO DE DATOS Y MUESTRA DE RESULTADOS = #
# ================================================== #

# Hyperparámetros a modificar. MUY IMPORTANTE!
a = 0.05 # Valor de alfa.
epochs = 2000 # Epochs.
lambda_reg = 0.2  # Valor de lambda para la tecnica de regularizacion L2.

# Inicia los parámetros del modelo con valores pequeños entre -0.01 y 0.01.
params = np.random.uniform(-0.01, 0.01, X_train.shape[1])

# Agrega los resultados de los errores y accuracy a listas vacías.
errors = []
train_accs, val_accs, test_accs = [], [], []

# Corre el loop de entrenamiento usando las funciones para cada epoch.
for epoch in range(epochs):
    params = GD(params, X_train, Y_train, a, lambda_reg)
    cost = cost_function(params, X_train, Y_train, lambda_reg) # Actualiza cost para la grafica de costo.
    errors.append(cost)

    train_accs.append(accuracy(params, X_train, Y_train))
    val_accs.append(accuracy(params, X_val, Y_val))
    test_accs.append(accuracy(params, X_test, Y_test))
        
# ================================================== #
# =============== CáLCULO DE MéTRICAS ============== #
# ================================================== #

# Regresa el valor del mean squared error de train, val, test.
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

# Regresa el valor de R^2 de train, val, test.
def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot)

# Calcula las metricas de ambos factores.
def calcular_metricas(theta, X, y):
    y_pred = h(theta, X) 
    return r2(y, y_pred), mse(y, y_pred)

r2_train, mse_train = calcular_metricas(params, X_train, Y_train)
r2_val, mse_val     = calcular_metricas(params, X_val, Y_val)
r2_test, mse_test   = calcular_metricas(params, X_test, Y_test)

# Calcular bias y varianza de forma numérica
final_train_acc = accuracy(params, X_train, Y_train)
final_val_acc   = accuracy(params, X_val, Y_val)
final_test_acc  = accuracy(params, X_test, Y_test)

bias_value = 1 - final_train_acc  # Error de entrenamiento como proxy del bias
varianza_value = abs(final_train_acc - final_val_acc)  # Diferencia entre train y val


# ================================================ #
# =========== ES HORA DE GRAFICAR! =============== #
# ================================================ #

# Gráficas unificadas en formato 2x2
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
X_axis = range(len(errors))

# Subplot 1: Matriz de Confusión
test_preds = predict(params, X_test)
cm = np.zeros((2,2), dtype=int)
for i in range(len(Y_test)):
    cm[Y_test[i], test_preds[i]] += 1
    
thresh = cm.max() / 2
for i in range(2):
    for j in range(2):
        axes[0,0].text(j, i, format(cm[i, j], "d"),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

im = axes[0,0].imshow(cm, interpolation="nearest", cmap=plt.cm.Greens)
axes[0,0].set_title(" Resultados en Matriz de Confusión")
axes[0,0].set_xlabel("Predicción")
axes[0,0].set_ylabel("Valor Real")
tick_marks = np.arange(2)
axes[0,0].set_xticks(tick_marks)
axes[0,0].set_xticklabels(["No Diabetes (0)", "Diabetes (1)"])
axes[0,0].set_yticks(tick_marks)
axes[0,0].set_yticklabels(["No Diabetes (0)", "Diabetes (1)"])
fig.colorbar(im, ax=axes[0,0]) # Barra de colores.

# Subplot 2: Valores verdaderos vs. Predicciones.
axes[0,1].scatter(range(len(Y_test)), Y_test, color="red", alpha=0.6, label="Valores Verdaderos")
axes[0,1].scatter(range(len(test_preds)), test_preds, color="blue", alpha=0.6, label="Predicciones")
axes[0,1].set_title("Valores verdaderos vs valores predecidos")
axes[0,1].set_xlabel("Pacientes")
axes[0,1].set_ylabel("(0 = No Diabetes, 1 = Diabetes)")
axes[0,1].legend()
axes[0,1].grid(True, linestyle="--", alpha=0.5)

# Subplot 3: Función de Costo
axes[1,0].plot(X_axis, errors, label="Cost", color="blue")
axes[1,0].set_title("Función de costo sobre epochs")
axes[1,0].set_xlabel("Epochs")
axes[1,0].set_ylabel("Cost")
axes[1,0].grid(True, linestyle="--", alpha=0.6)
axes[1,0].legend()

# Subplot 4: Accuracy de Train, Val, Test
def suavizar(valores, ventana=300):
    kernel = np.ones(ventana)/ventana
    return np.convolve(valores, kernel, mode="valid")

axes[1,1].plot(suavizar(train_accs), label="Train Accuracy", color="red")
axes[1,1].plot(suavizar(val_accs), label="Validation Accuracy", color="blue")
axes[1,1].plot(suavizar(test_accs), label="Test Accuracy", color="green")
axes[1,1].set_title("Accuracy de Train, Validation y Test Accuracy sobre epochs")
axes[1,1].set_xlabel("Epochs")
axes[1,1].set_ylabel("Accuracy")
axes[1,1].set_ylim(0.4, 1)
axes[1,1].grid(True, linestyle="--", alpha=0.6)
axes[1,1].legend()

# Desplegar valores finales.
fig.text(
    0.5, 0.01,
    f"Final Test Accuracy: {final_test_acc*100:.2f}%\n"
    f"Train: R²={r2_train:.3f}, MSE={mse_train:.3f} | "
    f"Val: R²={r2_val:.3f}, MSE={mse_val:.3f} | "
    f"Test: R²={r2_test:.3f}, MSE={mse_test:.3f}\n"
    f"Bias (aprox): {bias_value:.3f} | Varianza (aprox): {varianza_value:.3f}",
    ha="center", va="bottom",
    fontsize=16, fontweight="bold"
)

# Mostramos la gráfica.
plt.tight_layout(rect=[0, 0.07, 1, 1]) # Espacio para el título.
plt.show()