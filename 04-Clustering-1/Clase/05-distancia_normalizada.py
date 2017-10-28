import numpy as np
import pandas as pd
import os
from scipy.spatial import distance

if __name__ == '__main__':
    os.chdir('data')
    df = pd.read_csv("pima-indians-diabetes.data-small-orig", 
                 names = ['emb', 'gl2h', 'pad', 'ept', 'is2h', 'imc', 'fpd', 'edad', 'class'])
    md = distance.pdist(df.head(5), 'euclidean')
    print('\nLas distancias euclidianas para los datos de diabetes son:\n', md)
    mdn =  md/(1+md)
    print('\nLas distancias euclidianas normalizadas para los datos de diabetes son:\n', mdn)
    print('\nY las similaridades euclidianas correspondientes son:\n', 1-mdn)

    md2 = md/10000
    mdn2 =  md2/(1+md2)
    print('\nY para los datos pequeños:\n d:', mdn2, "\n s:",1-mdn2)