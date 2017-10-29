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
        leaf_font_size=14.,
    )
    plt.show()

if __name__ == '__main__':
    os.chdir('data')
    df = pd.read_csv("train_numbers.csv")

    # rellena los valores faltantes con la media
    df = df.fillna(df.mean())
    # obtiene el dendrograma con los ultimos 20 clusters
    centroid(df, 20)