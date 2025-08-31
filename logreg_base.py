# Grant Nathaniel Keegan

# Instalamos todas las dependencias.
import pandas as pd # Para trabajar y manipular los archivos csv.
import numpy as np # Para las fórmulas matemáticas de los algoritmos.
import matplotlib.pyplot as plt # Para plotear los resultados en tablas.
import csv # Para leer el archivo .csv con los datos.
import warnings # Evita que salgan warnings que ensucien la consola.
warnings.filterwarnings("ignore", category=FutureWarning)

#Importar los datos del archivo .csv
df = pd.read_csv('diabetes_data_upload.csv', header=0)
print(df.head(10))


print("Hay valores faltantes en la tabla?")
print(df.isnull().any())

# Reemplazar valores categóricos por binarios
df_modified = df.replace({
    "Yes": 1,"No": 0,
    "Positive": 1,"Negative": 0,
    "Male": 'M',"Female": 'F'
})

# Limpieza de datos | Convertiremos 'Yes' y 'No' en la tabla a binario.
# Guardar en un nuevo archivo CSV.
columnas = ['Age', 'Gender',
            'Polyuria', 'Polydipsia', 'sudden weight loss', 'weakness', 
            'Polyphagia', 'Genital thrush', 'visual blurring', 
            'Itching', 'Irritability', 'delayed healing', 
            'partial paresis', 'muscle stiffness', 'Alopecia', 'Obesity', 
            'class']

# Exportamos al nuevo archivo 'diabetes_data_modified.csv'.
df_modified.to_csv('diabetes_data_modified.csv', index=False)
df = df = pd.read_csv('diabetes_data_modified.csv', header=0)

class_counts = df["class"].value_counts()

plt.bar(class_counts.index.astype(str), class_counts.values, color=["green", "red"])
plt.title("Distribución de clases (Diabetes)")
plt.xlabel("Clase (0 = Negativo, 1 = Positivo)")
plt.ylabel("Número de pacientes")
plt.grid(axis='y')
plt.show()