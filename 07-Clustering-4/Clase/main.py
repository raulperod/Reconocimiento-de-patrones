# Inicializar el ambiente
import numpy as np
import pandas as pd
import os
import sys
from isodata import *

if __name__ == '__main__':
    np.set_printoptions(precision=2, suppress=True) # Cortar la impresión de decimales a 1
    os.chdir('data')
    LARGER_DISTANCE = sys.maxsize
    # Leer los datos de archivo
    df = pd.read_csv("datosProm.csv", names = ['A', 'B'])
    DATA_SET = df.values
    DATA_LEN = len(DATA_SET)
    isodata(DATA_SET, DATA_LEN, 0, LARGER_DISTANCE, 5, 3, 10, 5, 80, 2)
    