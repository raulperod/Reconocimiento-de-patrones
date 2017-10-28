import numpy as np
import pandas as pd
import os
from scipy.spatial import distance

if __name__ == '__main__':
    os.chdir('data')
    df = pd.read_csv("pima-indians-diabetes.data-small-orig", 
                 names = ['emb', 'gl2h', 'pad', 'ept', 'is2h', 'imc', 'fpd', 'edad', 'class'])

    # s(x, y) = 1 / d(x, y)             
    md = distance.pdist(df.head(5), 'euclidean')
    print('\nLas distancias euclidianas para los datos de diabetes son:\n', md)

    ms = 1/md
    print('\nLas similaridades euclidianas para los datos de diabetes son:\n', ms)

    md2 = md/10000
    print('\nLas similaridades euclidianas para los datos "pequeños" de diabetes son:\n', 1/md2)
    # s(x, y) = 1 / ( d(x, y) + 0.5 )
    ms = 1/(md+0.5)
    print('\nLas similaridades euclidianas para los datos de diabetes son:\n', ms)

    ms2 = 1/(md2+0.5)
    print('\nLas similaridades euclidianas para los datos "pequeños" de diabetes son:\n', ms2)