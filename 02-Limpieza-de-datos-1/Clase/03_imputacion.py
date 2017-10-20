import numpy as np
import pandas as pd
import os

def imputacion():
    #Lectura simple de datos
    df = pd.read_csv( 'pima-indians-diabetes.data-small', 
        names = ['emb', 'gl2h', 'pad', 'ept', 'is2h', 'imc', 'fpd', 'edad', 'class'] )
    # imprimo el data frame
    print(df, "\n")
    # los campos NaN, les pongo 0
    df2 = df.fillna(0)
    # imprimo el data frame 2
    print(df2, "\n")
    print (df2.describe())
    # Rellenar el valor NaN, con diferentes maneras
    print("Rellenando con el valor minimo \n", df.fillna(df.min()).describe(), "\n")
    print("Rellenando con el valor máximo \n", df.fillna(df.max()).describe(), "\n")
    print("Rellenando con la media \n", df.fillna(df.mean()).describe(), "\n")
    print("Rellenando con la mediana \n", df.fillna(df.median()).describe(), "\n")
    print("Rellenando con la moda \n", df.fillna(df.mode()).describe())
    # util en series de tiempo
    # Otra alternativa común es rellenar los valores faltantes con el valor no nulo previo o el siguiente:
    print(df, "\n")
    print("Replicar hacia enfrente\n", df.fillna(method='pad'), "\n")
    print("Replicar hacia atrás\n", df.fillna(method='bfill')
    # rellenar con el ultimo valor conocido
    print(df.fillna(method='pad', limit=1))

if __name__ == '__main__':
    #me muevo a la carpeta data
    os.chdir('data')
    imputacion()