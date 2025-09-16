# =================================================== #
# ==== Logistic Regression con XGBoost (MEJORADO) === #
# ============== Grant Nathaniel Keegan ============= #
# =================================================== #

# ================================================ #
# = INSTALACIóN Y CONFIGURACIóN DE DEPENDENCIAS == #
# ===== DEFINICIÓN DE VARIABLES Y PARÁMETROS ===== #
# ================================================ #

# Instalamos dependencias.
import pandas as pd # Para trabajar y manipular los archivos csv.
import numpy as np # Para las fórmulas matemáticas de los algoritmos. Ejemplo: np.dot: producto cruzado.
import math # Para operaciones como exp, log y sqrt.
import matplotlib.pyplot as plt # Para plotear los resultados en tablas.
import seaborn as sns # Para graficar la matríz de confusión.

# Dependencias de scikit learn.
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split # Divide los resultados.
from sklearn.preprocessing import StandardScaler # Escalamos los datos de X.
from sklearn.linear_model import LogisticRegression # Nuestro algoritmo principal de regresión logística.
from sklearn.model_selection import learning_curve # para graficar la curva de aprendizaje.
from sklearn.metrics import accuracy_score, classification_report, log_loss, confusion_matrix, mean_squared_error, r2_score

import xgboost as xgb # Nuestro algoritmo que utilizaremos para entrenar el modelo.

# Para ignorar warnings en la terminal.
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Leer el archivo modificado.
df = pd.read_csv("diabetes_data_modified.csv")

print(df.columns)

# Asignamos las columnas de la tabla a X y Y. También definimos los pesos del algoritmo (params).
X = df[["Age", "Gender", 
        "Polyuria", "Polydipsia", "sudden weight loss", "weakness","Polyphagia",
        "Genital thrush", "visual blurring", "Itching", "Irritability",
        "delayed healing", "partial paresis", "muscle stiffness", "Alopecia", "Obesity"
]]
Y = df["class"]  # Etiqueta binaria: si tiene diabetes o no.

# ================================================ #
# = DIVISION DE DATOS EN TRAIN, TEST, VALIDATION = #
# ================================================ #

# Dividimos los datos entre validación (train) (70%), prueba (test) (15%) y validación (15%).
# 364 pacientes prueba, 78 entrenamiento, 78 validacion.
# Con un random state de 42.
X_temp, X_test, Y_temp, Y_test = train_test_split(X, Y, test_size=0.15, random_state=42)
X_train, X_val, Y_train, Y_val = train_test_split(X_temp, Y_temp, test_size=0.176, random_state=42)

# Mostrar resultados de división entre train, test y val.
print(f"\n🏋🏻‍♂️ X_train: ({X_train.shape[0]} pacientes de 520)")
print(f"\n🧪 X_test:  ({X_test.shape[0]} pacientes de 520)")
print(f"\n✅ X_val:   ({X_val.shape[0]} pacientes de 520)")

print(f"\n🏋🏻‍♂️ Y_train: ({Y_train.shape[0]} pacientes de 520)")
print(f"\n🧪 Y_test:  ({Y_test.shape[0]} pacientes de 520)")
print(f"\n✅ Y_val:   ({Y_val.shape[0]} pacientes de 520)")

# Escalamiento de valores.
escalamiento = StandardScaler()
X_train_scaled = escalamiento.fit_transform(X_train)
X_test_scaled = escalamiento.fit_transform(X_test)
X_val_scaled = escalamiento.fit_transform(X_val)

# ================================================ #
# = ENTRENAMIENTO DE DATOS MUESTRA DE RESULTADOS = #
# ================================================ #

# Parametros de XGBoost.
params = {
    'objective': 'binary:logistic', # Clasificacion binaria.
    'eval_metric': ['logloss', 'error'], # Metricas para evaluar el modelo.
    'eta': 0.05, # Learning rate, similar a alpha en el modelo base.
    'max_depth': 8, # Profundidad máxima de cada árbol.
    'min_child_weight': 2, # Número de instancias en cada hoja
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'gamma': 0.1,
    'reg_lambda': 5, # Regularización estilo L2.
    'reg_alpha': 2 # Regularización estilo L1.
}

# Definimos epochs. MUY IMPORTANTE!
epochs = 300

# Crea nuestro modelo de XGBoost, y empieza
model = xgb.XGBClassifier(**params, n_estimators=300) # Creamos el modelo con los parametros.
eval_set = [(X_train_scaled, Y_train), (X_test_scaled, Y_test)]
model.fit(
    X_train_scaled, Y_train,
    eval_set=[(X_train_scaled, Y_train), (X_val_scaled, Y_val)],
    verbose=0
)
results = model.evals_result()

# Para calcular las métricas de predicción y accuracy que usaremos en nuestras gráficas.
train_predictions = model.predict(X_train_scaled)
test_predictions = model.predict(X_test_scaled)
train_accuracy = accuracy_score(Y_train, train_predictions)
test_accuracy = accuracy_score(Y_test, test_predictions)

# ================================================== #
# =============== CáLCULO DE MéTRICAS ============== #
# ================================================== #

# Calcular métricas finales
final_test_acc = test_accuracy
final_train_acc = train_accuracy

# R2 y MSE en cada dataset
r2_train = r2_score(Y_train, train_predictions)
mse_train = mean_squared_error(Y_train, train_predictions)

r2_val = r2_score(Y_val, model.predict(X_val_scaled))
mse_val = mean_squared_error(Y_val, model.predict(X_val_scaled))

r2_test = r2_score(Y_test, test_predictions)
mse_test = mean_squared_error(Y_test, test_predictions)

# Calculo de bias para metricas finales.
bias_value = np.mean(Y_train - train_predictions)

# Calculo de varianza para metricas finales.
varianza_value = abs(train_accuracy - test_accuracy)

# ================================================ #
# =========== ES HORA DE GRAFICAR! =============== #
# ================================================ #

# Crear la figura con 3 subplots en una fila.
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
X_axis = range(0, epochs)

# Subplot 1: Matríz de Confusión.
axes[0,0].set_title('Matriz de Confusión')
axes[0,0].set_ylabel('Predicción')
axes[0,0].set_xlabel('Valor Real')
cm = confusion_matrix(Y_test, test_predictions)
im = axes[0,0].imshow(cm, interpolation='nearest', cmap=plt.cm.Greens)
tick_marks = np.arange(len(np.unique(Y)))
axes[0,0].set_xticks(tick_marks)
axes[0,0].set_xticklabels(["No Diabetes (0)", "Diabetes (1)"])
axes[0,0].set_yticks(tick_marks)
axes[0,0].set_yticklabels(["No Diabetes (0)", "Diabetes (1)"])
for i, j in np.ndindex(cm.shape):
    axes[0,0].text(j, i, f"{cm[i, j]}")
fig.colorbar(im, ax=axes[0,0]) # Barra de colores.
    
# Subplot 2: Valores Verdaderos vs Predicciones
axes[0,1].set_title("Valores verdaderos vs valores predichos")
axes[0,1].set_xlabel("Pacientes")
axes[0,1].set_ylabel("(0 = No Diabetes, 1 = Diabetes)")
axes[0,1].scatter(range(len(Y_test)), Y_test, color="red", label="Valores Verdaderos")
axes[0,1].scatter(range(len(test_predictions)), test_predictions, color="blue", alpha=0.6, label="Predicciones")
axes[0,1].legend()

# Subplot 3: Función de Costo.
axes[1,0].set_title('Función de costo sobre epochs')
axes[1,0].set_xlabel('Epochs')
axes[1,0].set_ylabel('Log Loss')
axes[1,0].plot(X_axis, results['validation_0']['logloss'], color='red', label='Train Loss')
axes[1,0].plot(X_axis, results['validation_1']['logloss'], color='green', label='Test Loss')
axes[1,0].legend()
axes[1,0].grid(True, axis='y')

# Subplot 4: Accuracy de Train, Val, Test
def suavizar(valores, ventana=300):
    kernel = np.ones(ventana)/ventana
    return np.convolve(valores, kernel, mode="valid")

axes[1,1].set_title('Accuracy de Train y Test Accuracy sobre epochs')
axes[1,1].set_xlabel('Epochs')
axes[1,1].set_ylabel('Accuracy')

train_acc = 1 - np.array(results['validation_0']['error'])
test_acc  = 1 - np.array(results['validation_1']['error'])

train_acc_suav = suavizar(train_acc, ventana=20)
test_acc_suav  = suavizar(test_acc, ventana=20)
X_axis_suav = range(len(train_acc_suav))

axes[1,1].plot(X_axis_suav, train_acc_suav, color='red', label='Accuracy de entrenamiento')
axes[1,1].plot(X_axis_suav, test_acc_suav, color='green', label='Accuracy de prueba.')

axes[1,1].legend()
axes[1,1].grid(True, axis='y')

# Desplegar valores finales.
fig.text(
    0.5, 0.01,
    f"Final Test Accuracy: {final_test_acc*100:.2f}%\n"
    f"Train: R²={r2_train:.3f}, MSE={mse_train:.3f} | "
    f"Test: R²={r2_test:.3f}, MSE={mse_test:.3f}\n"
    f"Bias (aprox): {bias_value:.3f} | Varianza (aprox): {varianza_value:.3f}",
    ha="center", va="bottom",
    fontsize=12, fontweight="bold"
)

# Mostramos la gráfica.
plt.tight_layout(rect=[0, 0.07, 1, 1])
plt.show()