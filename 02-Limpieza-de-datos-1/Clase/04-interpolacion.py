import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os

def lineal(df, fig, axes):
    df['ept'].plot(ax=axes[0])
    df['ept'].plot(ax=axes[0], grid=True, kind="bar")

    dfi = df['ept'].interpolate()

    dfi.plot(ax=axes[1])
    dfi.plot(ax=axes[1], grid=True, kind="bar", color="red")
    df['ept'].plot(ax=axes[1], grid=True, kind="bar")
    print (df['ept'].interpolate())

def cuadradas(df, fig, axes):
    dfi = df['ept'].interpolate(method="quadratic")
    dfi.plot(ax=axes[0])
    dfi.plot(ax=axes[0], grid=True, kind="bar", color="red")
    df['ept'].plot(ax=axes[0], grid=True, kind="bar")
    print (df['ept'].interpolate())

def cubicas(df, fig, axes):
    
    dfi = df['ept'].interpolate(method="cubic")
    dfi.plot(ax=axes[1])
    dfi.plot(ax=axes[1], grid=True, kind="bar", color="red")
    df['ept'].plot(ax=axes[1], grid=True, kind="bar")
    print (df['ept'].interpolate())

if __name__ == '__main__':
    #me muevo a la carpeta data
    os.chdir('data')
    #Lectura simple de datos
    df = pd.read_csv( 'pima-indians-diabetes.data-small', 
        names = ['emb', 'gl2h', 'pad', 'ept', 'is2h', 'imc', 'fpd', 'edad', 'class'] )
    
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True, figsize=(10,10))
    
    lineal(df, fig, axes)
    cuadradas(df, fig, axes)
    cubicas(df, fig, axes)