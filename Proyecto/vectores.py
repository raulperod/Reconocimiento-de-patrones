import numpy as np
import pandas as pd
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC


def lineal(datos, etiqueta):
    X_trainPID, X_testPID, y_trainPID, y_testPID = train_test_split(
        datos.values, etiqueta.values.ravel(), test_size=0.1)

    c_valors = [0.1, 0.5, 1, 20, 100, 500, 1000]

    for i in c_valors:
        svmLineal = LinearSVC(C=i)
        start_time = time.time()
        svmLineal.fit(X_trainPID, y_trainPID)
        elapsed_time = time.time() - start_time

        preds_train_Lineal = svmLineal.predict(X_trainPID)
        fails_train_Lineal = np.sum(y_trainPID != preds_train_Lineal)

        preds_Lineal = svmLineal.predict(X_testPID)
        fails_Lineal = np.sum(y_testPID != preds_Lineal)

        print("SVM Lineal, C={} (default)\nPuntos mal clasificados (entrenamiento): {} de {} ({}%)\
            \nPuntos mal clasificados (prueba): {} de {} ({}%)\
            \nAciertos del {}%\nTiempo: {}\n\n"
            .format(i, fails_train_Lineal, len(y_trainPID), 100*fails_train_Lineal/len(y_trainPID),
                    fails_Lineal, len(y_testPID), 100*fails_Lineal/len(y_testPID), 
                    svmLineal.score(X_testPID, y_testPID)*100, elapsed_time))

def kernel(datos, etiqueta):
    X_trainPID, X_testPID, y_trainPID, y_testPID = train_test_split(
        datos.values, etiqueta.values.ravel(), test_size=0.1)

    c_valors = [0.1, 0.5, 1, 20, 100, 500, 1000]

    for i in c_valors:
        svmRbf = SVC(kernel='rbf', C=i)
        start_time = time.time()
        svmRbf.fit(X_trainPID, y_trainPID)
        elapsed_time = time.time() - start_time

        preds_train_Rbf = svmRbf.predict(X_trainPID)
        fails_train_Rbf = np.sum(X_trainPID != preds_train_Rbf)

        preds_Rbf = svmRbf.predict(X_testPID)
        fails_Rbf = np.sum(y_testPID != preds_Rbf)

        print("SVM RBF, C={} \nPuntos mal clasificados (entrenamiento): {} de {} ({}%)\
            \nPuntos mal clasificados (prueba): {} de {} ({}%)\
            \nAciertos del {}%\nTiempo: {}\n"
            .format(i, fails_train_Rbf, len(y_trainPID), 100*fails_train_Rbf/len(y_trainPID),
                    fails_Rbf, len(y_testPID), 100*fails_Rbf/len(y_testPID), 
                    svmRbf.score(X_testPID, y_testPID)*100, elapsed_time))

def kernel_gamma(datos, etiqueta):
    X_trainPID, X_testPID, y_trainPID, y_testPID = train_test_split(
        datos.values, etiqueta.values.ravel(), test_size=0.1)

    gamma_valors = [0.1, 0.5, 1, 20, 100, 500, 1000]

    for i in gamma_valors:
        svmRbf = SVC(kernel='rbf', C=1, gamma=i)
        start_time = time.time()
        svmRbf.fit(X_trainPID, y_trainPID)
        elapsed_time = time.time() - start_time

        preds_train_Rbf = svmRbf.predict(X_trainPID)
        fails_train_Rbf = np.sum(y_trainPID != preds_train_Rbf)

        preds_Rbf = svmRbf.predict(X_testPID)
        fails_Rbf = np.sum(y_testPID != preds_Rbf)

        print("SVM RBF, C=1, gamma={} \nPuntos mal clasificados (entrenamiento): {} de {} ({}%)\
            \nPuntos mal clasificados (prueba): {} de {} ({}%)\
            \nAciertos del {}%\nTiempo: {}\n"
            .format(i, fails_train_Rbf, len(y_trainPID), 100*fails_train_Rbf/len(y_trainPID),
                    fails_Rbf, len(y_testPID), 100*fails_Rbf/len(y_testPID), 
                    svmRbf.score(X_testPID, y_testPID)*100, elapsed_time))

def kernel_sigmoide(datos, etiqueta):
    X_trainPID, X_testPID, y_trainPID, y_testPID = train_test_split(
        datos.values, etiqueta.values.ravel(), test_size=0.1)

    svmSgm = SVC(kernel='sigmoid')
    start_time = time.time()
    svmSgm.fit(X_trainPID, y_trainPID)
    elapsed_time = time.time() - start_time

    preds_train_Sgm = svmSgm.predict(X_trainPID)
    fails_train_Sgm = np.sum(y_trainPID != preds_train_Sgm)

    preds_Sgm = svmSgm.predict(X_testPID)
    fails_Sgm = np.sum(y_testPID != preds_Sgm)

    print("SVM Sigmoide, C=1.0\nPuntos mal clasificados (entrenamiento): {} de {} ({}%)\
        \nPuntos mal clasificados (prueba): {} de {} ({}%)\
        \nAciertos del {}%\nTiempo: {}\n"
        .format(fails_train_Sgm, len(y_trainPID), 100*fails_train_Sgm/len(y_trainPID),
                fails_Sgm, len(y_testPID), 100*fails_Sgm/len(y_testPID), 
                svmSgm.score(X_testPID, y_testPID)*100, elapsed_time))

if __name__ == '__main__':

    os.chdir('datos')

    df = pd.read_csv("weather_limpio2.csv")
    etiqueta = df['Eventos']
    datos = df.drop('Eventos', 1)

    #lineal(datos, etiqueta)
    #kernel(datos, etiqueta)
    #kernel_gamma(datos, etiqueta)
    kernel_sigmoide(datos, etiqueta)