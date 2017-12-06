# Inicializar el ambiente
import numpy as np
import pandas as pd
import math
import random
import time
import os
import sys
from scipy.spatial import distance
from sklearn import cluster
from matplotlib import pyplot as plt
#%matplotlib inline

if __name__ == '__main__':
    np.set_printoptions(precision=2, suppress=True) # Cortar la impresión de decimales a 1

    os.chdir('datos')
    # Leer los datos de archivo, separar training y test y calcular "prototipos de clase"
    df = pd.read_csv("weather_limpio.csv")

    #test_point = df.loc[len(df)-1].values 
    #train_set = df.drop([len(df)-1]).values

    test_point = df.loc[0].values 
    train_set = df.drop([0]).values


    print("Datos de entrenaiento: \n{}\n\nDato de prueba:\n{}\n".format(train_set, test_point))

    num_clusters = 2
    k_means = cluster.KMeans(n_clusters=num_clusters, init="random")
    k_means.fit(train_set) 

    print("Prototipos de clase (centroides):\n", k_means.cluster_centers_)

    fig = plt.figure(figsize=(8, 5))

    colors = ['#ff0000', '#00ff00', '#0000ff', '#ff00ff', '#00ffff', '#ffff00', '#f6ff00', 
            '#2f800f', '#a221b5', '#21b5ac', '#b1216c']

    LARGER_DISTANCE = sys.maxsize

    k_neighs = 45 # 5 vecinos... aunque tomaremos sólo el más cercano
    neighbors_dists = [LARGER_DISTANCE] * k_neighs
    neighbors = [0] * k_neighs
    
    for i in range(len(train_set)):
        dist = distance.euclidean(train_set[i], test_point)
        for k in range(k_neighs):
            if (dist < neighbors_dists[k]) :
                for j in range(k_neighs-1, k, -1):
                    neighbors_dists[j] = neighbors_dists[j-1]
                    neighbors[j] = neighbors[j-1] 
                neighbors_dists[k] = dist
                neighbors[k] = i
                break
                
    print("Los {} vecinos más próximos son:".format(k_neighs))

    for k in range(k_neighs):
        clase = k_means.labels_[neighbors[k]]
        print("Vecino {}: {}, dist={}, clase={}, centroide={}"
            .format(k, neighbors[k], neighbors_dists[k], 
                    clase, k_means.cluster_centers_[clase]))
    print("\nEl nuevo punto es asignado a la clase", k_means.labels_[neighbors[0]])

    for i in range(num_clusters):
        dist = distance.euclidean(k_means.cluster_centers_[i], test_point)
        print ("Distancia del punto de prueba al prototipo de la clase {}: {}".format(i, dist))

    simple_vote = [0] * num_clusters
    winner = 0 

    for k in range(k_neighs):
        clase = k_means.labels_[neighbors[k]]
        simple_vote[clase] += 1

    for k in range(num_clusters):
        if (simple_vote[k] == max(simple_vote)):
            winner = k

    print("Votación simple:\nEl nuevo punto es asignado a la clase {} con {} vecinos cercanos.\n"
        .format(winner, simple_vote[winner]))

    print("Los {} vecinos más próximos y sus pesos ponderados son:".format(k_neighs))

    suma_dists = sum(neighbors_dists)
    neighbors_weights = [0] * k_neighs
    weighted_vote = [0] * num_clusters
    winner = 0 

    for k in range(k_neighs):
        neighbors_weights[k] = 1 - neighbors_dists[k] / suma_dists
        clase = k_means.labels_[neighbors[k]]
        weighted_vote[clase] += neighbors_weights[k]
        print("Vecino {}: peso={}, clase: {}"
            .format(k, neighbors_weights[k], k_means.labels_[neighbors[k]]))

    for k in range(num_clusters):
        if (simple_vote[k] == max(simple_vote)):
            winner = k

    print("\nVotación ponderada:")
    print("El nuevo punto es asignado a la clase {} con una votación de {}."
        .format(winner, weighted_vote[winner]))