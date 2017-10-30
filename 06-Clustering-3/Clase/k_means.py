import numpy as np
import pandas as pd
import math
import random
import time
from scipy.spatial import distance

# Definir una clase para expresar puntos y su asignación a un cluster
class DataPoint:
    def __init__(self, p):
        self.value = p[:]
        
    def set_value(self, p):
        self.value = p
    
    def get_value(self):
        return self.value
    
    def set_cluster(self, cluster):
        self.cluster = cluster
    
    def get_cluster(self):
        return self.cluster

def initialize_dataset(data, data_set, data_len):
    for i in range(data_len):
        point = DataPoint(data_set[i])
        point.set_cluster(None)
        data.append(point)
    return

def initialize_centroids(centroids, sampling_method, num_clusters, data_set, data_len):
    # print("Centroides inicializados en:")
    
    for c in range(num_clusters):
        if(sampling_method == 0):
            which = random.randint(0, data_len-1)
        elif(sampling_method == 1):
            which = c
        else:
            which = data_len-1-c
                
        centroids.append(list(data_set[which]))
    # imprimo los centroids elegidos
    # print(centroids)        
    return  

def update_clusters(centroids, num_clusters, data , data_len, larger_distance):
    changed = False
    
    for i in range(data_len):
        minDistance = larger_distance
        currentCluster = 0
        
        for j in range(num_clusters):
            dist = distance.euclidean(data[i].get_value(), centroids[j])
            if(dist < minDistance):
                minDistance = dist
                currentCluster = j
        
        if(data[i].get_cluster() is None or data[i].get_cluster() != currentCluster):
            data[i].set_cluster(currentCluster)
            changed = True
                
    return changed

def update_centroids(centroids, num_clusters, data, data_set, data_len):    
    # print("Los nuevos centroids son:")

    for j in range(num_clusters):
        means = [0] * data_set.shape[1]
            
        clusterSize = 0
        for k in range(data_len):
            if(data[k].get_cluster() == j):
                p = data[k].get_value()
                for i in range(data_set.shape[1]):
                    means[i] += p[i]
                clusterSize += 1

        if(clusterSize > 0):
            for i in range(data_set.shape[1]):
                centroids[j][i] = means[i] / clusterSize

    # print(centroids)        
    return    

def k_means( num_clusters, sampling_method, data_set, data_len, larger_distance):
    LARGER_DISTANCE = larger_distance
    # Leer los datos de archivo
    DATA_SET = data_set
    # Tamaño del conjunto de datos
    DATA_LEN = data_len
    # --------------------------
    # Crear el conjunto de datos
    data = []
    initialize_dataset(data, DATA_SET, DATA_LEN)
    # 1 - definir el numero de clusters
    NUM_CLUSTERS = num_clusters
    # 2 - Definir forma de muestreo; 0 = random, 1=head, 2=tail
    SAMPLING_METHOD = sampling_method
    centroids = []
    initialize_centroids(centroids, SAMPLING_METHOD, NUM_CLUSTERS, DATA_SET, DATA_LEN)
    # 3 Asignar cada punto del conjunto de datos al cluster donde la 
    # distancia del punto al centroide es menor.
    # Actualizar los clusters
    KEEP_WALKING = update_clusters(centroids, NUM_CLUSTERS, data, DATA_LEN, LARGER_DISTANCE) 
    # 4 - Calcular los centroides a partir de los puntos en cada cluster.
    # Actualizar los centroides
    update_centroids(centroids, NUM_CLUSTERS, data, DATA_SET, DATA_LEN)
    # 5 - Repetir los pasos 3 y 4 hasta que no haya cambios en los clusters.
    while(KEEP_WALKING):
        KEEP_WALKING = update_clusters(centroids, NUM_CLUSTERS, data, DATA_LEN, LARGER_DISTANCE) 
        if (KEEP_WALKING):
            update_centroids(centroids, NUM_CLUSTERS, data, DATA_SET, DATA_LEN)
        else :  
            members = [0] * NUM_CLUSTERS
            for i in range(DATA_LEN):
                members[data[i].get_cluster()] += 1

            for j in range(NUM_CLUSTERS):
                print(f"\nCluster {j}: {members[j]} miembros.")
                print(np.asarray(centroids[j]))