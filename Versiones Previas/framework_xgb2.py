# Grant Nathaniel Keegan | A01700753 | logreg_base.py
# Aplica un modelo de regresión logística CON FRAMEWORKS para predecir
# si futuros pacientes sufren de diabetes o no.

# ================================================ #
# = INSTALACIóN Y CONFIGURACIóN DE DEPENDENCIAS == #
# ================================================ #

# Instalamos dependencias.
import pandas as pd # Para trabajar y manipular los archivos csv.
import numpy as np # Para las fórmulas matemáticas de los algoritmos. Ejemplo: np.dot: producto cruzado.
import math # Para operaciones como exp, log y sqrt.
import matplotlib.pyplot as plt # Para plotear los resultados en tablas.
import seaborn as sns # Para graficar la matríz de confusión.
from sklearn.linear_model import SGDClassifier
import xgboost as xgb

# Dependencias de scikit learn.
from sklearn.model_selection import train_test_split # Divide los resultados.
from sklearn.preprocessing import StandardScaler # Escalamos los datos de X.
from sklearn.linear_model import LogisticRegression # Nuestro algoritmo principal de regresión logística.
from sklearn.model_selection import learning_curve # para graficar la curva de aprendizaje.
from sklearn.metrics import accuracy_score, classification_report, log_loss, confusion_matrix

# Para ignorar warnings en la terminal.
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Leer el archivo modificado.
df = pd.read_csv("diabetes_data_modified.csv")

# ================================================ #
# ===== DEFINICIÓN DE VARIABLES Y PARÁMETROS ===== #
# ================================================ #

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

# Dividimos los datos entre validación (train) (60%), prueba (test) (20%) y validación (20%).
# test_size = 0.2. El 20% de los datos se usarán para test. 416 pacientes prueba, 102, validacion.
# Con un random state de 42.
X_temp, X_test, Y_temp, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
X_train, X_val, Y_train, Y_val = train_test_split(X_temp, Y_temp, test_size=0.25, random_state=42)

# Mostrar resultados de división entre train, test y val.
print(f"\n🏋🏻‍♂️ X_train: ({X_train.shape[0]} pacientes de 520)")
print(f"\n🧪 X_test:  ({X_test.shape[0]} pacientes de 520)")
print(f"\n✅ X_val:   ({X_val.shape[0]} pacientes de 520)")

print(f"\n🏋🏻‍♂️ Y_train: ({Y_train.shape[0]} pacientes de 520)")
print(f"\n🧪 Y_test:  ({Y_test.shape[0]} pacientes de 520)")
print(f"\n✅ Y_val:   ({Y_val.shape[0]} pacientes de 520)")

escalamiento = StandardScaler()
X_train_scaled = escalamiento.fit_transform(X_train)
X_test_scaled = escalamiento.fit_transform(X_test)
X_val_scaled = escalamiento.fit_transform(X_val)

# ================================================ #
# = ENTRENAMIENTO DE DATOS MUESTRA DE RESULTADOS = #
# ================================================ #

# Parametros de XGBoost.
params = {
    'objective': 'multi:softmax',
    'num_class': 4,
    'reg_alpha': 1,
    'reg_lambda': 1,
    'eta': 0.1,
    'eval_metric': ['mlogloss', 'merror']
}

model = xgb.XGBClassifier(**params) # Creamos el modelo con los parametros.
eval_set = [(X_train_scaled, Y_train), (X_test_scaled, Y_test)]
model.fit(X_train_scaled, Y_train, eval_set=eval_set, verbose=True)
results = model.evals_result()

epochs = len(results['validation_0']['mlogloss'])
x_axis = range(0, epochs)

# ================================================ #
# =========== ES HORA DE GRAFICAR! =============== #
# ================================================ #

# Crear la figura con 3 subplots en una fila.
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].plot(x_axis, results['validation_0']['mlogloss'], label='Train Loss')
axes[0].plot(x_axis, results['validation_1']['mlogloss'], label='Test Loss')
axes[0].legend()
axes[0].set_xlabel('Epochs')
axes[0].set_ylabel('Log Loss')
axes[0].set_title('Train and Test Loss over Epochs')
axes[0].grid(True)

axes[1].plot(x_axis, 1 - np.array(results['validation_0']['merror']), label='Train Accuracy')
axes[1].plot(x_axis, 1 - np.array(results['validation_1']['merror']), label='Test Accuracy')
axes[1].legend()
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Train and Test Accuracy over Epochs')
axes[1].grid(True)

train_predictions = model.predict(X_train_scaled)
test_predictions = model.predict(X_test_scaled)

train_accuracy = accuracy_score(Y_train, train_predictions)
test_accuracy = accuracy_score(Y_test, test_predictions)

conf_matrix = confusion_matrix(Y_test, test_predictions)
im = axes[2].imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
axes[2].set_title('Confusion Matrix')

tick_marks = np.arange(len(np.unique(Y)))
axes[2].set_xticks(tick_marks)
axes[2].set_xticklabels(np.unique(Y))
axes[2].set_yticks(tick_marks)
axes[2].set_yticklabels(np.unique(Y))

thresh = conf_matrix.max() / 2
for i, j in np.ndindex(conf_matrix.shape):
    axes[2].text(j, i, f"{conf_matrix[i, j]}",
                 horizontalalignment="center",
                 color="white" if conf_matrix[i, j] > thresh else "black")

axes[2].set_ylabel('True label')
axes[2].set_xlabel('Predicted label')

# Agregar la barra de colores
fig.colorbar(im, ax=axes[2])

plt.tight_layout()
plt.show()

print(f"Final Train Accuracy: {train_accuracy * 100:.2f}%")
print(f"Final Test Accuracy: {test_accuracy * 100:.2f}%")