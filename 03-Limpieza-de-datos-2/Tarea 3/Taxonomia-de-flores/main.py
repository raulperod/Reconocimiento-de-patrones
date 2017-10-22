import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import warnings 

warnings.filterwarnings("ignore")

def valores_faltantes(df, names):
    print('Descripcion de los datos')
    print(df.describe())
    print('Porcentaje de datos nulos por columna')
    for name in names:
        print(name, df[name].isnull().sum() / df.shape[0] * 100)

def generar_diagrama_caja(df, col):
    df.boxplot(column=col)
    plt.show()

def generar_diagramas_de_caja(df, column_names):
    for name in column_names:
        generar_diagrama_caja(df, name)

if __name__ == '__main__':
    os.chdir('data')
    column_names =  ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    df = pd.read_csv('iris.data', names = column_names)
    # elimino la columna class ya que no se usara
    column_names.remove('class')
    valores_faltantes(df, column_names)
    # realizar imputacion con la mediana
    df = df.fillna(df.median())
    # generar diagramas de caja
    generar_diagramas_de_caja(df, column_names)