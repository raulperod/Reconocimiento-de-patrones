import pandas as pd
import os

if __name__ == '__main__':
    
    os.chdir('data')
    df = pd.read_csv( 'auto-mpg.csv',
                      names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model-year', 'origin', 'car-name'] )
    
    print(df.head(3))
    print(df.info())
    print(df.describe())
    