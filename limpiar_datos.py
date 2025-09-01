# Grant Nathaniel Keegan | A01700753 | lipmiar datos.py
# Toma el archivo .csv original (diabetes_data_upload.csv)
# y lo convierte a diabetes_data_modified.csv

# Instalamos dependencias.
import pandas as pd # Para trabajar y manipular los archivos csv.
import csv # Para leer el archivo .csv con los datos.
import matplotlib.pyplot as plt # Para graficar los datos iniciales.
from matplotlib.patches import Patch # Para los colores de la tabla.
import numpy as np # Para la gráfica de datos iniciales.
import warnings # Evita que salgan warnings que ensucien la consola.
warnings.filterwarnings("ignore", category=FutureWarning)

#Importar los datos del archivo .csv
df = pd.read_csv('diabetes_data_upload.csv', header=0)
print("TABLA INICIAL")
print(df.head(10)) # Despliega la primera tabla

print("Hay valores faltantes en la tabla?")
print(df.isnull().any())

# Reemplazar valores categóricos por binarios
df_modified = df.replace({
    "Yes": 1,"No": 0,
    "Positive": 1,"Negative": 0,
    "Male": '1',"Female": '0'
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
print("TABLA MODIFICADA")
print(df.head(10)) # Despliega la SEGUNDA tabla

# ========== GRAFICAR DATOS INICIALES (SOLO SÍNTOMAS Y GÉNERO )============ #

# Muestra gráfica de los datos de cada paciente (1 o 0):
# Contar cuántos pacientes tienen 0 y 1 por cada columna
conteo_0 = []
conteo_1 = []

for i in columnas:
    valores = df[i].value_counts()
    conteo_0.append(valores.get(0, 0))
    conteo_1.append(valores.get(1, 0))

# Configurar posiciones en X
x = np.arange(len(columnas))
width = 0.35

# Crear la gráfica de barras
fig, grafica1 = plt.subplots(figsize=(14, 6))

# Colores personalizados según columna
colores_0 = []
colores_1 = []
for col in columnas:
    if col == 'class':
        colores_0.append('blue') # Negativo a diabetes
        colores_1.append('red') # Positivo a diabetes
    elif col == 'Gender':
        colores_0.append('pink') # Mujer
        colores_1.append('lightblue') # Hombre
    else:
        colores_0.append('green') # No a síntoma
        colores_1.append('orange') # Sí a síntoma

# Dibujar barras
bars1 = grafica1.bar(x - width/2, conteo_0, width, label='Valor = 0', color=colores_0)
bars2 = grafica1.bar(x + width/2, conteo_1, width, label='Valor = 1', color=colores_1)

grafica1.set_xlabel('Síntomas y Resultado')
grafica1.set_ylabel('Cantidad de pacientes')
grafica1.set_title('Muestra inicial de datos (diabetes_data_modified)')
grafica1.set_xticks(x)
grafica1.set_xticklabels(columnas, rotation=45, ha="right")

# Añadir leyenda personalizada

leyenda_colores = [
    Patch(color='blue', label='Negativo a diabetes'),
    Patch(color='red', label='Class = 1 Positivo a diabetes'),
    Patch(color='pink', label='Mujer'),
    Patch(color='lightblue', label='Hombre'),
    Patch(color='green', label='No a síntoma'),
    Patch(color='orange', label='Sí a síntoma'),
]

grafica1.legend(handles=leyenda_colores, title="Variables")
fig.tight_layout()
plt.show()