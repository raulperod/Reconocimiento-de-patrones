import numpy as np
import pandas as pd
from IPython.display import Image, display  
from sklearn import tree
import pydotplus # brew install graphviz, pip install pydotplus
from io import StringIO
from IPython.display import Image, display
# Para hacer un muestreo aleatorio
from sklearn.model_selection import train_test_split

if __name__ == '__main__':

    classes_names = ['Nada', 'Neblina', 'Nieve', 'Lluvia', 'Neblina-Lluvia', 'Tornado', 
        'Lluvia-Nieve', 'Lluvia-Tormenta', 'Neblina-Lluvia-Tormenta', 'Lluvia-Granizo-Tormenta']

    feats_names = ['TemperaturaMaxC', 'TemperaturaPromC', 'TemperaturaPromC', 'HumedadMax', 
        'HumedadProm', 'HumedadMin', 'VientoMaxKMXH', 'VientoPromKMXH', 'PrecipitacionMM']

    df = pd.read_csv("datos/weather_limpio2.csv")

    etiqueta = df['Eventos']
    datos = df.drop('Eventos', 1)

    train_features, test_features, train_targets, test_targets = train_test_split(
        datos.values, etiqueta.values.ravel(), test_size=0.1)

    test_targets = list(test_targets)
    train_targets = list(train_targets)

    print ("Clases de la muestra de prueba: ", test_targets)

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(train_features, train_targets)

    dotfile = StringIO()
    tree.export_graphviz(clf, out_file=dotfile, class_names=classes_names, feature_names=feats_names,
                            filled=False, rounded=True)

    graph = pydotplus.graph_from_dot_data(dotfile.getvalue())
    graph.write_png('arbol.png')    
    #display(Image(graph.create_png()))

    predictions_test = clf.predict(test_features)
    fails_test = np.sum(test_targets != predictions_test)
    print("Objetivos: ", test_targets)
    print("Resultados: ", list(predictions_test))
    print("Puntos mal clasificados en el conjunto de prueba: {} de {} ({}%)\n"
        .format(fails_test, len(test_targets), 100*fails_test/len(test_targets)))