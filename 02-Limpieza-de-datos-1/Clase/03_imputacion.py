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

if __name__ == '__main__':
    #me muevo a la carpeta data
    os.chdir('data')
    imputacion()