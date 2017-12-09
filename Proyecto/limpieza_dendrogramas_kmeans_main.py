import numpy as np
import pandas as pd
import os
import sys
from limpieza import obtener_valores_faltantes, realizar_imputacion_mediana, generar_diagrama_caja
from dendrograma import single, complete, centroid
from kmeans import k_means
#from isodata import isodata
from sklearn import cluster

def diagramas_de_caja(df, names):
    for name in names:
        if name != 'Eventos':
            generar_diagrama_caja(df, name)

if __name__ == '__main__':
    
    np.set_printoptions(precision=2, suppress=True) # Cortar la impresión de decimales a 1
    os.chdir('datos')
    
    """ # Fase de limpieza
    df = pd.read_csv('weather.csv')
    # limpieza del conjunto de datos
    df = obtener_valores_faltantes(df)
    #print(df.describe())
    df = realizar_imputacion_mediana(df)
    #print(df.describe())
    df.to_csv('weather_limpio2.csv', index=False)
    #diagramas_de_caja(df, df.columns.get_values().tolist())
    """ 
    
    """ Dendrogramas
    single(df, 4)
    complete(df, 4)
    centroid(df, 4)
    """
    """ Kmeans
    df = pd.read_csv('weather_limpio.csv').drop('Eventos', axis=1)
    LARGER_DISTANCE = sys.maxsize
    # Leer los datos de archivo
    # rellena los valores faltantes con la media
    DATA_SET = df.values
    # Tamaño del conjunto de datos
    DATA_LEN = len(DATA_SET)
    # inicializa el k means
    # num_clusters, sampling_method, data_set, data_len, larger_distance
    k_means( num_clusters=12, sampling_method=0, data_set=DATA_SET
        , data_len=DATA_LEN, larger_distance=LARGER_DISTANCE)   
    """

    