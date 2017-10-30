import numpy as np
import pandas as pd
import os
import sys
from k_means import *

if __name__ == '__main__':
    np.set_printoptions(precision=2, suppress=True) # Cortar la impresión de decimales a 1
    os.chdir('data')
    LARGER_DISTANCE = sys.maxsize
    # Leer los datos de archivo
    DATA_SET = pd.read_csv("datosProm.csv", names = ['A', 'B']).values
    # Tamaño del conjunto de datos
    DATA_LEN = len(DATA_SET)
    # inicializa el k means
    k_means( 3, 0, DATA_SET, DATA_LEN, LARGER_DISTANCE)   