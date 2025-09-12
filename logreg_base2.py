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
df_x = df[["Age", "Gender", 
        "Polyuria", "Polydipsia", "sudden weight loss", "weakness","Polyphagia",
        "Genital thrush", "visual blurring", "Itching", "Irritability",
        "delayed healing", "partial paresis", "muscle stiffness", "Alopecia", "Obesity"
]]
df_y = df["class"]  # Etiqueta binaria: si tiene diabetes o no.

# Split into training, validation, and test
train_size = int(0.6 * len(df_x))
val_size = int(0.2 * len(df_x))
test_size = len(df_x) - train_size - val_size

train_x, val_x, test_x = df_x[:train_size], df_x[train_size:train_size + val_size], df_x[train_size + val_size:]
train_y, val_y, test_y = df_y[:train_size], df_y[train_size:train_size + val_size], df_y[train_size + val_size:]

# Add bias
train_x = np.c_[np.ones(len(train_x)), train_x]
val_x = np.c_[np.ones(len(val_x)), val_x]
test_x = np.c_[np.ones(len(test_x)), test_x]

train_y = np.array(train_y)
val_y = np.array(val_y)
test_y = np.array(test_y)

num_classes = len(np.unique(train_y))
params_list = np.random.uniform(-0.01, 0.01, (num_classes, train_x.shape[1]))

# Hypothesis
def h(params, sample):
    z = -np.dot(params, sample)
    return 1 / (1 + np.exp(z))

# Cost function (cross entropy)
def cost_function(params, samples, y, current_class):
    targets = (y == current_class).astype(int)
    predictions = h(params, samples.T)
    cost = -targets * np.log(predictions) - (1 - targets) * np.log(1 - predictions)
    return np.mean(cost)

# Gradient Descent
def GD(params, samples, y, alfa, current_class):
    targets = (y == current_class).astype(int)
    predictions = h(params, samples.T)
    errors = predictions - targets
    gradient = np.dot(errors, samples) / len(samples)
    params -= alfa * gradient
    
    cost = cost_function(params, samples, y, current_class)
    return params, cost

# Prediction 
def predict_class(params_list, sample):
    probabilities = h(params_list, sample.T)
    return np.argmax(probabilities, axis=0)

# Calculate Accuracy
def calculate_accuracy(params_list, samples, labels, current_class=None):
    predictions = predict_class(params_list, samples)
    if current_class is None:
        return np.mean(predictions == labels)
    else:
        relevant_samples = labels == current_class
        if not relevant_samples.any():
            return 0
        return np.mean(predictions[relevant_samples] == labels[relevant_samples])

# Training
alfa = 0.3
max_epochs = 2000

errors = []
train_accuracies = {i: [] for i in range(num_classes)}
val_accuracies = {i: [] for i in range(num_classes)}
test_accuracies = {i: [] for i in range(num_classes)}
costs_per_class = {i: [] for i in range(num_classes)}
overall_train_accuracy = []
overall_val_accuracy = []
overall_test_accuracy = []

for epoch in range(max_epochs):
    epoch_costs = []
    for current_class in range(num_classes):
        params_list[current_class], class_cost = GD(params_list[current_class], train_x, train_y, alfa, current_class)
        epoch_costs.append(class_cost)
        costs_per_class[current_class].append(class_cost)
        
        train_acc = calculate_accuracy(params_list, train_x, train_y, current_class)
        val_acc = calculate_accuracy(params_list, val_x, val_y, current_class) 
        test_acc = calculate_accuracy(params_list, test_x, test_y, current_class)
        
        train_accuracies[current_class].append(train_acc)
        val_accuracies[current_class].append(val_acc)
        test_accuracies[current_class].append(test_acc)
    
    epoch_cost = np.mean(epoch_costs)
    errors.append(epoch_cost)

    overall_train_acc = calculate_accuracy(params_list, train_x, train_y)
    overall_val_acc = calculate_accuracy(params_list, val_x, val_y)
    overall_test_acc = calculate_accuracy(params_list, test_x, test_y)
    
    overall_train_accuracy.append(overall_train_acc)
    overall_val_accuracy.append(overall_val_acc)
    overall_test_accuracy.append(overall_test_acc)

    if (epoch + 1) % 500 == 0: 
        print(f"Epoch {epoch+1}:")
        for current_class in range(num_classes):
            print(f"  Class {current_class}: Train Accuracy = {train_accuracies[current_class][-1]:.4f}, Validation Accuracy = {val_accuracies[current_class][-1]:.4f}, Test Accuracy = {test_accuracies[current_class][-1]:.4f}")

final_val_accuracy = calculate_accuracy(params_list, val_x, val_y)
print(f"Final Validation Accuracy: {final_val_accuracy * 100:.2f}%")

# Graphs
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(errors)
plt.title('Cost Function Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Cost')

plt.subplot(1, 2, 2)
plt.plot(overall_train_accuracy, label='Train Accuracy')
plt.plot(overall_val_accuracy, label='Validation Accuracy')
plt.plot(overall_test_accuracy, label='Test Accuracy')
plt.title('Overall Train, Validation, and Test Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

for current_class in range(num_classes):
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    
    # Accuracy 
    axs[0].plot(train_accuracies[current_class], label=f'Train Accuracy Class {current_class}', linestyle='--')
    axs[0].plot(val_accuracies[current_class], label=f'Validation Accuracy Class {current_class}', linestyle='--')
    axs[0].plot(test_accuracies[current_class], label=f'Test Accuracy Class {current_class}')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_title(f'Accuracy over Epochs for Class {current_class}')
    axs[0].legend()

    # Cost 
    axs[1].plot(costs_per_class[current_class], label=f'Cost for Class {current_class}')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Cost')
    axs[1].set_title(f'Cost over Epochs for Class {current_class}')
    axs[1].legend()

    plt.tight_layout()
    plt.show()

# Final Test Predictions
test_predictions = predict_class(params_list, test_x)

# Final Test Accuracy
final_test_accuracy = calculate_accuracy(params_list, test_x, test_y)
print(f"Final Test Accuracy: {final_test_accuracy * 100:.2f}%")

# Confusion Matrix
confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

for i in range(len(test_y)):
    true_label = test_y[i]
    pred_label = test_predictions[i]
    confusion_matrix[true_label, pred_label] += 1

plt.figure(figsize=(8, 6))
plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(num_classes)
plt.xticks(tick_marks, [0, 1, 2, 3]) 
plt.yticks(tick_marks, [0, 1, 2, 3])

thresh = confusion_matrix.max() / 2.0
for i in range(num_classes):
    for j in range(num_classes):
        plt.text(j, i, format(confusion_matrix[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if confusion_matrix[i, j] > thresh else "black")

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
