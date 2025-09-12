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
from xgboost import XGBClassifier # Para optimizar el modelo usando xgboost.

# Dependencias de scikit learn.
from sklearn.model_selection import train_test_split # Divide los resultados.
from sklearn.preprocessing import StandardScaler # Escalamos los datos de X.
from sklearn.linear_model import LogisticRegression # Nuestro algoritmo principal de regresión logística.
from sklearn.model_selection import learning_curve # para graficar la curva de aprendizaje.
from sklearn.metrics import accuracy_score, classification_report, log_loss, confusion_matrix
from sklearn.linear_model import SGDClassifier

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
# ============ ESCALAMIENTO DE DATOS ============= #
# ================================================ #

# Dividimos los datos entre validación (train) (70%), prueba (test) (20%) y validación (10%).
# test_size = 0.2. El 20% de los datos se usarán para test. 416 pacientes prueba, 102, validacion.
# Con un random state de 42.
X_temp, X_test, Y_temp, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
X_train, X_val, Y_train, Y_val = train_test_split(X_temp, Y_temp, test_size=1/3, random_state=42)

# Mostrar resultados de división entre train, test y val.
print(f"\n🏋🏻‍♂️ X_train: ({X_train.shape[0]} pacientes de 520)")
print(f"\n🧪 X_test:  ({X_test.shape[0]} pacientes de 520)")
print(f"\n✅ X_val:   ({X_val.shape[0]} pacientes de 520)")

print(f"\n🏋🏻‍♂️ Y_train: ({Y_train.shape[0]} pacientes de 520)")
print(f"\n🧪 Y_test:  ({Y_test.shape[0]} pacientes de 520)")
print(f"\n✅ Y_val:   ({Y_val.shape[0]} pacientes de 520)")

# Vamos a escalar los datos de x.

escalamiento = StandardScaler()
X_train_scaled = escalamiento.fit_transform(X_train)
X_test_scaled = escalamiento.fit_transform(X_test)
X_val_scaled = escalamiento.fit_transform(X_val)

# ================================================ #
# = ENTRENAMIENTO DE DATOS MUESTRA DE RESULTADOS = #
# ================================================ #

# Guardamos las perdidas en cada epoca.
train_losses, val_losses, test_losses = [], [], []
epochs = 100 # Definimos epochs. MUY IMPORTANTE!

# Creamos el loop para entrenar la regresion logistica 100 veces.
for epoch in range(epochs):
    model = LogisticRegression(max_iter=100, solver='lbfgs')
    model.fit(X_train_scaled, Y_train)

    # Probabilidades para log_loss
    Y_train_proba = model.predict_proba(X_train_scaled)
    Y_val_proba = model.predict_proba(X_val_scaled)
    Y_test_proba = model.predict_proba(X_test_scaled)

    # Calculamos pérdidas agregandolas a las listas.
    train_losses.append(log_loss(Y_train, Y_train_proba))
    val_losses.append(log_loss(Y_val, Y_val_proba))
    test_losses.append(log_loss(Y_test, Y_test_proba))

# Calcula el cambio de presicion al aumentar las iteraciones.
train_sizes, train_scores, val_scores = learning_curve(
    estimator=LogisticRegression(max_iter=100),
    X=X_train_scaled,
    y=Y_train,
    train_sizes=np.linspace(0.2, 1.0, 8), # Eje x de la curva de aprendizaje.
    cv=5, # Utiliza cross validation.
    scoring='accuracy',
    shuffle=True,
    random_state=42
)

# Promedia los resultados de entrenamiento y precision.
train_accuracies = np.mean(train_scores, axis=1)
val_accuracies = np.mean(val_scores, axis=1)

# Accuracy fijo del set de prueba
final_model = LogisticRegression(max_iter=100)
final_model.fit(X_train_scaled, Y_train)
Y_pred = final_model.predict(X_test_scaled)
test_acc = final_model.score(X_test_scaled, Y_test)
test_accuracies = [test_acc] * len(train_sizes)
    
# ================================================ #
# =========== ES HORA DE GRAFICAR! =============== #
# ================================================ #

# Crear la figura con 3 subplots en una fila
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
X_axis = range(0, epochs) # Rango para graficar.

# Subplot 1: Perdidas sobre epochs.
axes[0].plot(X_axis, train_losses, label='Train Loss', color='blue')
axes[0].plot(X_axis, val_losses, label='Validation Loss', color='purple')
axes[0].plot(X_axis, test_losses, label='Test Loss', color='green', linestyle='--')
axes[0].set_title("Loss por Epoch (Logistic Regression)")
axes[0].set_xlabel("Epochs")
axes[0].set_ylabel("Log Loss")
axes[0].legend()
axes[0].grid(True, axis='y')

# Subplot 2: Curva de aprendizaje de train y test.
axes[1].plot(train_sizes * len(X_train_scaled), train_accuracies, label='Entrenamiento', marker='o')
axes[1].plot(train_sizes * len(X_train_scaled), val_accuracies, label='Validación', marker='s')
axes[1].plot(train_sizes * len(X_train_scaled), test_accuracies, label='Prueba', linestyle='--', marker='^', color='green')
axes[1].set_title("Curva de aprendizaje (Logistic Regression)")
axes[1].set_xlabel("Tamaño del set de entrenamiento")
axes[1].set_ylabel("Precisión (Accuracy)")
axes[1].set_ylim(0.7, 1.1)
axes[1].legend()
axes[1].grid(True, axis='y')

# Subplot 3: Matríz de Confusión.
cm = confusion_matrix(Y_test, Y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False, ax=axes[2])
axes[2].set_title('Matriz de Confusión')
axes[2].set_xlabel('Predicción')
axes[2].set_ylabel('Valor Real')

# Mostramos la gráfica entera.
plt.tight_layout()
plt.show()