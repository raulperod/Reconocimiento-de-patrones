import numpy as np
import pandas as pd
import os

def primera_descripcion(df):
    print( 'Primera descripcion\n')
    print( df.describe(), '\n' )
    print( df, '\n' )

def segunda_descripcion(df):
    print( 'Segunda descripcion\n')
    print ('Tabla de valores nulos')
    print (df.isnull(), '\n')

    print ('Contabilidad de valores nulos por columna')
    print (df.isnull().sum(), '\n')

    print ('Porcentaje de datos nulos por columna')
    # calculo los porcentajes
    emb_null_pje = df['emb'].isnull().sum() / df.shape[0] * 100
    gl2h_null_pje = df['gl2h'].isnull().sum() / df.shape[0] * 100
    pad_null_pje = df['pad'].isnull().sum() / df.shape[0] * 100
    ept_null_pje = df['ept'].isnull().sum() / df.shape[0] * 100
    is2h_null_pje = df['is2h'].isnull().sum() / df.shape[0] * 100
    imc_null_pje = df['imc'].isnull().sum() / df.shape[0] * 100
    fpd_null_pje = df['fpd'].isnull().sum() / df.shape[0] * 100
    edad_null_pje = df['edad'].isnull().sum() / df.shape[0] * 100

    print('emb', emb_null_pje)
    print('gl2h', gl2h_null_pje)
    print('pad', pad_null_pje)
    print('ept', ept_null_pje)
    print('is2h', is2h_null_pje)
    print('imc', imc_null_pje)
    print('fpd', fpd_null_pje)
    print('edad', edad_null_pje)

def tercera_descripcion(df):
    # imprime los datos
    print(df, '\n')
    # remplaza los ceros en ept por valores nulos
    df.loc[df['ept'] == 0,'ept'] = np.nan
    # imprime los datos de nuevo
    print(df)
    # descripcion de los nuevos datos
    print(df.info(), '\n')
    print(df.describe(), '\n')

    print(f'Suma y promedio de ept: ({df["ept"].sum()}, {df["ept"].mean()})\n')
    print(f'Promedio tomando en cuenta los 0s: {df["ept"].sum()/20}\n')

def valores_faltantes():
    #Lectura simple de datos
    df = pd.read_csv( 'pima-indians-diabetes.data-small', 
        names = ['emb', 'gl2h', 'pad', 'ept', 'is2h', 'imc', 'fpd', 'edad', 'class'] )
    # primera descripcion
    primera_descripcion(df)
    # segunda descripcion
    segunda_descripcion(df)
    # nuevo analisis
    df2 = pd.read_csv("pima-indians-diabetes.data-small-orig", 
                 names = ['emb', 'gl2h', 'pad', 'ept', 'is2h', 'imc', 'fpd', 'edad', 'class'])
    # tercera descripcion             
    tercera_descripcion(df2)

if __name__ == '__main__':
    #me muevo a la carpeta data
    os.chdir('data')
    valores_faltantes()