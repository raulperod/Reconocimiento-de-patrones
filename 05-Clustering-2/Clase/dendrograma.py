import numpy as np
import pandas as pd
import os
from scipy.spatial import distance
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings 

warnings.filterwarnings("ignore")

def single(df, numero_clusters):
    Z = linkage(df, 'single')
    plt.figure(figsize=(12, 5))
    dendrogram(
        Z,
        truncate_mode='lastp',  # mostrar sólo los últims p clusters
        p=numero_clusters,                   
        show_leaf_counts=True,  # mostrar entre paréntesis el número de elementos en cada cluster
        leaf_font_size=14.,
    )
    plt.show()

def complete(df, numero_clusters):
    Z = linkage(df, 'complete')
    plt.figure(figsize=(12, 5))
    dendrogram(
        Z,
        truncate_mode='lastp',  # mostrar sólo los últims p clusters
        p=numero_clusters,                   
        show_leaf_counts=True,  # mostrar entre paréntesis el número de elementos en cada cluster
        leaf_font_size=14.,
    )
    plt.show()

def centroid(df, numero_clusters):
    Z = linkage(df, 'centroid')
    plt.figure(figsize=(12, 5))
    dendrogram(
        Z,
        truncate_mode='lastp',  # mostrar sólo los últims p clusters
        p=numero_clusters,                   
        show_leaf_counts=True,  # mostrar entre paréntesis el número de elementos en cada cluster
        leaf_font_size=14.,
    )
    plt.show()

if __name__ == '__main__':
    os.chdir('data')
    df = pd.read_csv("pima-indians-diabetes.data", 
                 names = ['emb', 'gl2h', 'pad', 'ept', 'is2h', 'imc', 'fpd', 'edad', 'class'])
    # rellena los valores faltantes con la media
    df.loc[df['pad'] == 0,'pad'] = df['pad'].mean()
    df.loc[df['ept'] == 0,'ept'] = df['ept'].mean()
    df.loc[df['is2h'] == 0,'is2h'] = df['is2h'].mean()
    df.loc[df['imc'] == 0,'imc'] = df['imc'].mean()

    x = df.head(10)

    np.set_printoptions(precision=1, suppress=True) # Cortar la impresión de decimales a 1
    # Convertir el vector de distancias a una matriz cuadrada
    md = distance.squareform(distance.pdist(x, 'euclidean')) 
    print(md)

    #single(x, 5)
    #complete(x, 5)
    #centroid(x, 5)
    centroid(df, 10)
    centroid(df, 4)