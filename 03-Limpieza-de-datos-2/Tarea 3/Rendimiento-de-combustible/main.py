import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import warnings 

warnings.filterwarnings("ignore")

def valores_faltantes(df):
    print('Porcentaje de datos nulos por columna')
    names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model-year', 'origin', 'car-name']
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
    df = pd.read_csv( 'auto-mpg.csv', 
        names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model-year', 'origin', 'car-name'] )
    
    valores_faltantes(df)
    # realizar imputacion con la mediana
    df = df.fillna(df.median())
    # generar diagramas de caja
    column_names =  ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration']
    generar_diagramas_de_caja(df, column_names)