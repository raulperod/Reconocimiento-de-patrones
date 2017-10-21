import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import warnings 

warnings.filterwarnings("ignore")

def diagrama_caja_con_valores_faltantes(df):
    print(df) # imprime el data frame
    print(df.describe()) # da una descripcion del data frame
    # hace un diagrama de caja de la columna gl2h
    df.boxplot(column='gl2h') 
    plt.show()
    # hace un diagrama de caja con todas las columnas
    df.boxplot() 
    plt.show() 

def diagrama_caja_con_imputacion(df):
    df2 = df.fillna(df.mean()) # rellena los valores faltantes con la media
    print(df2) # imprime el data frame
    print(df2.describe(), '\n') # describe los datos del data frame
    # hace un diagrama de caja del data frame
    df2.boxplot() 
    plt.show()

if __name__ == '__main__':
    os.chdir('data')
    df = pd.read_csv("pima-indians-diabetes.data-small-orig", 
                 names = ['emb', 'gl2h', 'pad', 'ept', 'is2h', 'imc', 'fpd', 'edad', 'class'])

    df2 = pd.read_csv("pima-indians-diabetes.data-small", 
                 names = ['emb', 'gl2h', 'pad', 'ept', 'is2h', 'imc', 'fpd', 'edad', 'class'])

    #diagrama_caja_con_valores_faltantes(df)
    diagrama_caja_con_imputacion(df2)