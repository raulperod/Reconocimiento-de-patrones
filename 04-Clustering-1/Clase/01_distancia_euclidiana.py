import numpy as np
import pandas as pd
import os
from scipy.spatial import distance

def ejemplo_simple():
    a = (1,2,3)
    b = (4,5,6)
    print(f'La distancia entre el punto {a} y el punto {b} es: {distance.euclidean(a,b)}')

def diabetes(df):
    print(df)
    md = distance.pdist(df, 'euclidean')
    print(f'\nLas distancias euclidianas ({md.size}) para los datos de diabetes son: {md}\n')

if __name__ == '__main__':
    ejemplo_simple()
    os.chdir('data')
    df = pd.read_csv("pima-indians-diabetes.data-small-orig", 
                 names = ['emb', 'gl2h', 'pad', 'ept', 'is2h', 'imc', 'fpd', 'edad', 'class'])
    diabetes(df)

