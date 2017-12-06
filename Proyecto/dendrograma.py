import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

def single(df, numero_clusters):
    Z = linkage(df, 'single')
    plt.figure(figsize=(12, 5))
    dendrogram( 
        Z,
        truncate_mode='lastp',  # mostrar sólo los últims p clusters
        show_leaf_counts=True,
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
        show_leaf_counts=True,
        p=numero_clusters,                   
        leaf_font_size=14.,
    )
    plt.show()

def centroid(df, numero_clusters):
    Z = linkage(df, 'centroid')
    plt.figure(figsize=(12, 5))
    dendrogram(Z, truncate_mode='lastp', p=numero_clusters, show_leaf_counts=True, leaf_font_size=14. )
    plt.show()
