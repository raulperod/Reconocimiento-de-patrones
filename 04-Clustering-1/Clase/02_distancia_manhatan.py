import numpy as np
import pandas as pd
import os
from scipy.spatial import distance

if __name__ == '__main__':
    os.chdir('data')
    df = pd.read_csv("pima-indians-diabetes.data-small-orig", 
                 names = ['emb', 'gl2h', 'pad', 'ept', 'is2h', 'imc', 'fpd', 'edad', 'class'])
    md_manhattan = distance.pdist(df, 'cityblock')
    print(f'\nLas distancias del uber ({md_manhattan.size}) para los datos de diabetes son: {md_manhattan}\n')