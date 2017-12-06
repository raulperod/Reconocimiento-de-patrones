# Inicializar el ambiente
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn import cluster # Auxiliar
import os
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#%matplotlib inline

def gauss(df, df_pure, df_class):
    cut = df.shape[0] // 3 * 2

    complete_set = df_pure.values
    complete_targets = df_class.values.ravel()
    train_set = df_pure.head(cut).values
    train_targets = df_class.head(cut).values.ravel()
    test_set = df_pure.tail(df.shape[0]-cut).values
    test_targets = df_class.tail(df.shape[0]-cut).values.ravel()

    clf = GaussianNB()
    clf.fit(complete_set, complete_targets)
    class_complete = clf.predict(complete_set)
    fails_complete = np.sum(complete_targets  != class_complete)
    print("Puntos mal clasificados en el conjunto completo: {} de {} ({}%)\n"
        .format(fails_complete, len(complete_set), 100*fails_complete/len(complete_set)))

    clf.fit(train_set, train_targets)
    class_predict_train = clf.predict(train_set)
    fails_train = np.sum(train_targets  != class_predict_train)
    print("Puntos mal clasificados en el conjunto de entrenamiento: {} de {} ({}%)\n"
        .format(fails_train, len(train_set), 100*fails_train/len(train_set)))

    class_predict_test = clf.predict(test_set)
    fails_test = np.sum(test_targets  != class_predict_test)
    print("Puntos mal clasificados en el conjunto de prueba: {} de {} ({}%)\n"
        .format(fails_test, len(test_set), 100*fails_test/len(test_set)))



if __name__ == "__main__":

    np.set_printoptions(precision=2, suppress=True) # Cortar la impresión de decimales a 1

    os.chdir('datos')
    df = pd.read_csv("weather_limpio2.csv")

    df_pure = df[list(['TemperaturaMaxC', 'TemperaturaPromC', 'TemperaturaMinC', 'HumedadMax', 
        'HumedadProm', 'HumedadMin', 'VientoMaxKMXH', 'VientoPromKMXH', 'PrecipitacionMM'])]

    df_class = df[list(['Eventos'])]

    gauss(df, df_pure, df_class)

