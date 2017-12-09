import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
    
def perceptron():
    # Importar datos de Iris Data Set
    df = pd.read_csv("datos/weather_limpio.csv")
    # Etiqueta de clase de cada vector ejemplo
    Y = df['Eventos'].values
    # Vector de características
    X = df.drop('Eventos', 1).values
    # Datos para entrenameinto y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.5)
    # Tasa de aprendizaje
    eta=0.01

    # Vector de pesos inicial
    w = np.zeros(X_train.shape[1])
    # Entrenamiento
    for xi, target in zip(X_train, y_train):
        activation = np.dot(xi, w)
        output = np.where(activation >= 0.0, 1, 0)
        error = target - output
        w += eta * error * xi
    # Prueba
    errores = 0
    for xi, target in zip(X_test, y_test) :
        activation = np.dot(xi, w)
        output = np.where(activation >= 0.0, 1, 0)
        if (target != output) :
            errores += 1
    print("{} vectores mal clasificados de {} ({}%)".format(errores, len(y_test), errores/len(y_test)*100))

    # 2a ronda de entrenamiento
    w = np.zeros(X_train.shape[1])
    vectores = 0
    shuffled_data = shuffle(list(zip(X_train, y_train)))
    for xi, target in shuffled_data:
        activation = np.dot(xi, w)
        output = np.where(activation >= 0.0, 1, 0)
        error = target - output
        w += eta * error * xi
        if (target != output) :
            vectores += 1
    print("{} vectores de entrenamiento mal clasificados de {} ({}%)".
        format(vectores, len(y_train), vectores/len(y_train)*100))

    # Vector de pesos inicial
    w = np.zeros(X.shape[1])
    # 3a ronda de entrenamiento... con todos los datos
    vectores = 0
    for xi, target in zip(X, Y):
        activation = np.dot(xi, w)
        output = np.where(activation >= 0.0, 1, 0)
        error = target - output
        w += eta * error * xi
        if (target != output) :
            vectores += 1
    print("{} vectores de entrenamiento mal clasificados de {} ({}%)".
        format(vectores, len(Y), vectores/len(Y)*100))
        
    # Prueba
    errores = 0
    for xi, target in zip(X_test, y_test) :
        activation = np.dot(xi, w)
        output = np.where(activation >= 0.0, 1, 0)
        if (target != output) :
            errores += 1
    print("{} vectores mal clasificados de {} ({}%)".format(errores, len(y_test), 
                                                            errores/len(y_test)*100))

def adaline():
    # Importar datos de Iris Data Set
    df = pd.read_csv("datos/weather_limpio.csv")
    # Etiqueta de clase de cada vector ejemplo
    Y = df['Eventos'].values
    # Vector de características
    X = df.drop('Eventos', 1).values
    # Tasa de aprendizaje
    eta=0.01
    # Re-etiquetar las clase de cada vector ejemplo en [-1,1]
    yAd = np.where(Y == 0, -1, 1)
    # Normalizar los vectores de características
    XAd = (X - X.mean()) / X.std()
    # Número de iteraciones
    niter = 15
    # Vector de pesos inicial
    wAd = np.zeros(XAd.shape[1] + 1)
    # Number of misclassifications
    errors = []
    # Cost function
    costs = []
    train_dataAd = XAd[:50]
    test_dataAd = XAd[50:]
    train_yAd = yAd[:50]
    test_yAd = yAd[50:]
    # Entrenamiento
    for i in range(niter):
        output = np.dot(train_dataAd, wAd[1:]) + wAd[0]
        errors = train_yAd - output
        wAd[1:] += eta * train_dataAd.T.dot(errors)
        wAd[0] += eta * errors.sum()
    # Prueba
    errores = 0
    for xi, target in zip(test_dataAd, test_yAd) :
        activation = np.dot(xi, wAd[1:]) + wAd[0]
        output = np.where(activation >= 0.0, 1, -1)
        if (target != output) :
            errores += 1
    print("{} vectores mal clasificados de {} ({}%)".format(errores, len(test_dataAd), 
                                                            errores/len(test_dataAd)*100))

def backpropagation():
    # Importar datos de Iris Data Set
    df = pd.read_csv("datos/weather_limpio.csv")
    # Etiqueta de clase de cada vector ejemplo
    Y = df['Eventos'].values
    # Vector de características
    X = df.drop('Eventos', 1).values
    X_trainff, X_testff, y_trainff, y_testff = train_test_split(X, Y, test_size=.5)
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, activation='logistic', #identity  tanh
                        hidden_layer_sizes=(4,4,2), random_state=1, 
                        learning_rate_init=0.001, max_iter=5000)

    clf.fit(X_trainff, y_trainff)                         

    # Prueba
    errores = 0
    for xi, target in zip(X_testff, y_testff) :
        output = clf.predict(xi.reshape(1, -1))
        if (target != output) :
            errores += 1
    print("{} vectores mal clasificados de {} ({}%)".format(errores, len(X_testff), 
                                                            errores/len(X_testff)*100))

if __name__ == '__main__':
    #perceptron()
    #adaline()
    backpropagation()