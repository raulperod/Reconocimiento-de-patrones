import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings 
warnings.filterwarnings("ignore")

def obtener_valores_faltantes(df):
    df.loc[df['VientoMaxKMXH'] == 0,'VientoMaxKMXH'] = np.nan
    df.loc[df['VientoPromKMXH'] == 0,'VientoPromKMXH'] = np.nan

    print ('Contabilidad de valores faltantes por columna')
    print (df.isnull().sum(), '\n')

    print ('Porcentaje de valores faltantes por columna')
    # calculo los porcentajes
    tmax_null_pje = df['TemperaturaMaxC'].isnull().sum() / df.shape[0] * 100
    tprom_null_pje = df['TemperaturaPromC'].isnull().sum() / df.shape[0] * 100
    tmin_null_pje = df['TemperaturaMinC'].isnull().sum() / df.shape[0] * 100
    hmax_null_pje = df['HumedadMax'].isnull().sum() / df.shape[0] * 100
    hprom_null_pje = df['HumedadProm'].isnull().sum() / df.shape[0] * 100
    hmin_null_pje = df['HumedadMin'].isnull().sum() / df.shape[0] * 100
    vmax_null_pje = df['VientoMaxKMXH'].isnull().sum() / df.shape[0] * 100
    vprom_null_pje = df['VientoPromKMXH'].isnull().sum() / df.shape[0] * 100
    prec_null_pje = df['PrecipitacionMM'].isnull().sum() / df.shape[0] * 100
    
    print('TemperaturaMaxC', tmax_null_pje)
    print('TemperaturaPromC', tprom_null_pje)
    print('TemperaturaMinC', tmin_null_pje)
    print('HumedadMax', hmax_null_pje)
    print('HumedadProm', hprom_null_pje)
    print('HumedadMin', hmin_null_pje)
    print('VientoMaxKMXH', vmax_null_pje)
    print('VientoPromKMXH', vprom_null_pje)
    print('PrecipitacionMM', prec_null_pje)
    
    return df

def realizar_imputacion_mediana(df):
    #print("Rellenando con la media \n", df.fillna(df.mean()).describe(), "\n")
    #print("Rellenando con la mediana \n", df.fillna(df.median()).describe(), "\n")
    #print("Rellenando con la moda \n", df.fillna(df.mode()).describe(), '\n')
    return df.fillna(df.median())

def realizar_interpolacion(df):
    df_gl2h = df['gl2h'].interpolate()
    df_pad = df['pad'].interpolate()
    df_ept = df['ept'].interpolate()
    df_is2h = df['is2h'].interpolate()
    df_imc = df['imc'].interpolate()

    print('gl2h\n', df_gl2h.describe())
    print('pad\n', df_pad.describe())
    print('ept\n', df_ept.describe())
    print('is2h\n', df_is2h.describe())
    print('imc\n', df_imc.describe())

def generar_diagrama_caja(df, col):
    df.boxplot(column=col)
    plt.show()