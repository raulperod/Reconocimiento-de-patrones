import pandas as pd
import os

if __name__ == '__main__':
    #me muevo a la carpeta data
    os.chdir('data')
    #Lectura simple de datos
    df = pd.read_csv( 'pima-indians-diabetes.data', names = ['emb', 'gl2h', 'pad', 'ept', 'is2h', 'imc', 'fpd', 'edad', 'class'] )
    # informacion
    print(df.head(10))
    print( df.info())
    print( df.describe() )