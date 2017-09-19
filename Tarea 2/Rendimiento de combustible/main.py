import pandas as pd
import os

if __name__ == '__main__':
    #me muevo a la carpeta data
    os.chdir('data')
    #Lectura simple de datos
    df = pd.read_csv( 'auto-mpg.csv',
                      names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model-year', 'origin', 'car-name'] )
    # informacion
    print(df.head(3))
    print(df.info())
    print(df.describe())
    