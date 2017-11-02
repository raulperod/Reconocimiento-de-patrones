# Inicializar el ambiente
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

def initialize_centroids(centroids, data_set, data_len, num_clusters, sampling_method):

    for c in range(num_clusters):
        if (sampling_method == 0) :
            which = random.randint(0, data_len-1)
        elif (sampling_method == 1):
            which = c
        else :
            which = data_len-1-c
                
        centroids.append(list(data_set[which]))
        
    return

def update_clusters(centroids, data, data_len, larger_distance, num_clusters, n_min, elim, members):
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
    
    members = [0] * num_clusters
    for i in range(data_len):
        members[data[i].get_cluster()] += 1

    # Marcar los grupos a eliminar      
    to_eliminate = []
    for j in range(num_clusters):
        if (members[j] < n_min):
            to_eliminate.append(j)
                
    # Eliminar grupos marcados y recorrer índices
    elim = 0
    is_null = 0
    for j in to_eliminate: # to_eliminate contiene los índices originales a eliminar...
        # Puesto que se han eliminado 'elim' clusters, los índices se han recorrido 
        # 'elim' veces; el verdero ínidice a eliminar es j_actual = j-elim
        j_actual = j-elim
        for i in range(data_len):
            cluster = data[i].get_cluster()
            if (cluster == j_actual):
                data[i].set_cluster(None)
                is_null += 1
            elif (cluster != None and cluster > j_actual) :
                data[i].set_cluster(cluster-1)
        elim += 1
        for i in range(j_actual, num_clusters-1):
            members[i] = members[i + 1]
    
    num_clusters -= elim

    return changed, elim, members

def update_centroids(centroids, data, data_set, data_len, num_clusters ):
    centroids = []

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
                means[i] = means[i] / clusterSize
            centroids.append(means)
    
    return centroids

def update_deltas(deltas, delta, centroids, data, data_len, num_clusters, members):
    deltas = [0] * num_clusters
    delta = 0
    
    for i in range(data_len):
        cluster = data[i].get_cluster()
        deltas[cluster] += distance.euclidean(data[i].get_value(), centroids[cluster])
    mem = 0
    for i in range(num_clusters):
        delta += deltas[i]
        mem += members[i]
        deltas[i] /= members[i]
       
    delta /= mem
    return delta, deltas

def divide_clusters(centroids, data, data_set, data_len, num_clusters, k_init, n_min, s_max, larger_distance, elim, members):
    # Cálculo de desviaciones estandar
    sigma_vect = [[0] * data_set.shape[1]] * num_clusters
    for d in range(data_len):
        cluster = data[d].get_cluster()
        p = data[d].get_value()
        for i in range(data_set.shape[1]):
            sigma_vect[cluster][i] += (p[i] - centroids[cluster][i])**2        
    candidates = []
    for cluster in range(num_clusters):
        for i in range(data_set.shape[1]):
            sigma_vect[cluster][i] = math.sqrt(sigma_vect[cluster][i]) / members[cluster]
            if (sigma_vect[cluster][i] > s_max):
                candidates.append(cluster)
                break # Sucio... pero eficiente :-) ... ya encontramos un atributo con elevada sigma
    
    divided = False
    for cluster in candidates:
        cond = num_clusters < k_init/2 or (deltas[cluster] > delta and members[cluster] > 2 * n_min)
        if(cond) :
            centroids.pop(cluster)
            points = []
            for d in range(data_len):
                if (data[d].get_cluster() == cluster):
                    points.append(data[d].get_value())
            dist = distance.squareform(distance.pdist(points, 'euclidean'))
            idx = (dist==dist.max()).argmax()
            z1 = list(points[idx // len(points)])
            z2 = list(points[idx % len(points)])
            centroids.append(z1)
            centroids.append(z2)
            num_clusters += 1
            divided = True
    
    if (divided) :
        kw, elim, members = update_clusters(centroids, data, data_len, larger_distance, num_clusters, elim, members)
        centroids, update_centroids(centroids, data, data_set, data_len, num_clusters)
    
    return centroids, elim, members

def mix_clusters(centroids, data, data_set, data_len, num_clusters, larger_distance, l_min, elim, members):
    dist = distance.squareform(distance.pdist(centroids, 'euclidean'))
    flag = math.floor(dist.max() * 10)
    dist[dist == 0] = flag
    
    mixed = False
    while (dist.min() < flag):
        idx = (dist==dist.min()).argmax()
        z1 = idx // len(centroids)
        z2 = idx % len(centroids)
        
        if (dist.min() < l_min):
            dist[z1] = flag
            dist[:,z1] = flag
            dist[z2] = flag
            dist[:,z2] = flag
            z = [sum(x)/2 for x in zip(centroids[z1], centroids[z2])]
            centroids[z1] = z
            centroids[z2] = [larger_distance]*data_set.shape[1]
            num_clusters -= 1
            mixed = True
        else :
            dist[z1][z2] = flag
            dist[z2][z1] = flag
        
    if (mixed) :
        kw, elim, members = update_clusters(centroids, data, data_len, larger_distance, num_clusters, elim, members)
        centroids, update_centroids(centroids, data, data_set, data_len, num_clusters)

    return centroids, elim, members

def isodata(data_set, data_len, sampling_method, larger_distance, k_init, n_min, i_max, s_max, l_min, p_max):
    # --------------------------
    # Crear el conjunto de datos
    data = []
    initialize_dataset(data, data_set, data_len)
    num_clusters = k_init # valor de k
    iteration = 0
    # 2) Seleccionar arbitrariamente los centroides iniciales
    # Definir forma de muestreo; 0 = random, 1=head, 2=tail
    # SAMPLING_METHOD = sampling_method 
    centroids = []
    # --------------------------
    # Inicializar los centroides
    initialize_centroids(centroids, data_set, data_len, num_clusters, sampling_method)
    # 3) Asignar cada punto del conjunto de datos al cluster donde la distancia del punto al centroide es menor.
    # 4) Eliminar los clusters con menos de $n_{min}$ elementos. Ajustar el valor de $k$ y reetiquetar los clusters.
    # --------------------------
    # Actualizar los clusters
    elim, members = 0, []
    KEEP_WALKING, elim, members = update_clusters(centroids, data, data_len, larger_distance, num_clusters, n_min, elim, members)
    # 5) Recalcular los centroides a partir de los puntos actualmente en cada cluster. 
    # Si se eliminaron clusters en el paso 4) el algoritmo regresa al paso 3).
    # --------------------------
    # Actualizar los centroides
    centroids = update_centroids(centroids, data, data_set, data_len, num_clusters)
    while(elim > 0):
        KEEP_WALKING, elim, members = update_clusters(centroids, data, data_len, larger_distance, num_clusters, n_min, elim, members)
        centroids = update_centroids(centroids, data, data_set, data_len, num_clusters)
    # 6) Calcular las distancias promedio $\Delta_j$ de los puntos de un cluster 
    # a su centroide y la distancia promedio general $\Delta$.
    delta, deltas = 0, []
    delta, deltas = update_deltas(deltas, delta, centroids, data, data_len, num_clusters, members)
    # 7) Si esta es la última iteración, terminar. En caso contrario verificar 
    # si quedan la mitad o menos de los clusters iniciales y de ser así ir al 
    # paso 8 (dividir clusters). En caso contrario, si la iteración es par o 
    # el número de clusters es mayor que el doble de los clusters iniciales, 
    # entonces ir al paso 9 (unir). En caso contrario, volver al paso 3 (como $k$-means).
    # Ejecutar sólo desués de haber "activado" los pasos 8 y 9
    centroids, elim, members = divide_clusters(centroids, data, data_set, data_len, num_clusters, k_init, n_min, s_max, larger_distance, elim, members) # paso 8
    centroids, elim, members = mix_clusters(centroids, data, data_set, data_len, num_clusters, larger_distance, l_min, elim, members) # paso 9
    # Reproducido aquí para facilitar la ejecución
    iteration +=1
    while(iteration < i_max and KEEP_WALKING):
        if (num_clusters <= k_init / 2) :
            centroids, elim, members = divide_clusters(centroids, data, data_set, data_len, num_clusters, k_init, n_min, s_max, larger_distance, elim, members)
        elif (iteration % 2 == 0 or num_clusters > 2 * k_init) :
            centroids, elim, members = mix_clusters(centroids, data, data_set, data_len, num_clusters, larger_distance, l_min, elim, members)
            
        KEEP_WALKING = update_clusters(centroids, data, data_len, larger_distance, num_clusters, n_min, elim, members)
        if (KEEP_WALKING):
            centroids = update_centroids(centroids, data, data_set, data_len, num_clusters)
        else :
            members = [0] * num_clusters
            for i in range(data_len):
                members[data[i].get_cluster()] += 1

            for j in range(num_clusters):
                print(f"\nCluster {j}: {members[j]} miembros.")
                print(np.asarray(centroids[j]))

            print ("No más cambios.")
        iteration += 1

