import numpy as np
import pandas as pd
import os

def tratamiento_de_valores_faltantes():
    #Lectura simple de datos
    df = pd.read_csv( 'pima-indians-diabetes.data-small', 
        names = ['emb', 'gl2h', 'pad', 'ept', 'is2h', 'imc', 'fpd', 'edad', 'class'] )
    print(df, '\n')
    # elimina todos los renglones con valores faltantes
    print(df.dropna())
    # elimina los renglones que no tengan 8 columnas limpias
    print(df.dropna(thresh=8), '\n')
    # elimina los renglones que no tengan 7 columnas limpias
    print(df.dropna(thresh=7))

if __name__ == '__main__':
    #me muevo a la carpeta data
    os.chdir('data')
    tratamiento_de_valores_faltantes()