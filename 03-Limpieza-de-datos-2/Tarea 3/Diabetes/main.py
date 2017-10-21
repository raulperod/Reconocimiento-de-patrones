import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import warnings 

warnings.filterwarnings("ignore")

def obtener_valores_faltantes(df):
    # obtener los valores faltantes
    df.loc[df['gl2h'] == 0,'gl2h'] = np.nan
    df.loc[df['pad'] == 0,'pad'] = np.nan
    df.loc[df['ept'] == 0,'ept'] = np.nan
    df.loc[df['is2h'] == 0,'is2h'] = np.nan
    df.loc[df['imc'] == 0,'imc'] = np.nan

    return df

def generar_diagrama_caja(df, col):
    df.boxplot(column=col)
    plt.show()

if __name__ == '__main__':
    os.chdir('data')
    df = pd.read_csv("pima-indians-diabetes.data", 
                 names = ['emb', 'gl2h', 'pad', 'ept', 'is2h', 'imc', 'fpd', 'edad', 'class'])

    # obtengo los valores faltantes
    df = obtener_valores_faltantes(df)
    # realizar imputacion con la mediana
    df = df.fillna(df.median())
    # diagramas de caja por columna
    generar_diagrama_caja(df, 'gl2h')
    generar_diagrama_caja(df, 'pad')
    generar_diagrama_caja(df, 'ept')
    generar_diagrama_caja(df, 'is2h')
    generar_diagrama_caja(df, 'imc') 