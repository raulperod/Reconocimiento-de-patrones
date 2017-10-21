import numpy as np
import pandas as pd
import os

def obtener_valores_faltantes(df):

    # obtener los valores faltantes
    df.loc[df['gl2h'] == 0,'gl2h'] = np.nan
    df.loc[df['pad'] == 0,'pad'] = np.nan
    df.loc[df['ept'] == 0,'ept'] = np.nan
    df.loc[df['is2h'] == 0,'is2h'] = np.nan
    df.loc[df['imc'] == 0,'imc'] = np.nan

    print ('Contabilidad de valores nulos por columna')
    print (df.isnull().sum(), '\n')

    print ('Porcentaje de datos nulos por columna')
    # calculo los porcentajes
    emb_null_pje = df['emb'].isnull().sum() / df.shape[0] * 100
    gl2h_null_pje = df['gl2h'].isnull().sum() / df.shape[0] * 100
    pad_null_pje = df['pad'].isnull().sum() / df.shape[0] * 100
    ept_null_pje = df['ept'].isnull().sum() / df.shape[0] * 100
    is2h_null_pje = df['is2h'].isnull().sum() / df.shape[0] * 100
    imc_null_pje = df['imc'].isnull().sum() / df.shape[0] * 100
    fpd_null_pje = df['fpd'].isnull().sum() / df.shape[0] * 100
    edad_null_pje = df['edad'].isnull().sum() / df.shape[0] * 100
    
    print('emb', emb_null_pje)
    print('gl2h', gl2h_null_pje)
    print('pad', pad_null_pje)
    print('ept', ept_null_pje)
    print('is2h', is2h_null_pje)
    print('imc', imc_null_pje)
    print('fpd', fpd_null_pje)
    print('edad', edad_null_pje)

    return df

def realizar_imputacion(df):
    print("Rellenando con la media \n", df.fillna(df.mean()).describe(), "\n")
    print("Rellenando con la mediana \n", df.fillna(df.median()).describe(), "\n")
    print("Rellenando con la moda \n", df.fillna(df.mode()).describe(), '\n')

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


if __name__ == '__main__':
    #me muevo a la carpeta data
    os.chdir('data')
    # Analice los problemas de valores faltantes en el conjunto 
    # de datos Pima Indians Diabetes completo.
    
    # Leer csv de diabetes completo
    df = pd.read_csv( 'pima-indians-diabetes.data', 
        names = ['emb', 'gl2h', 'pad', 'ept', 'is2h', 'imc', 'fpd', 'edad', 'class'] )

    # obtengo un data frame con los valores faltantes
    df = obtener_valores_faltantes(df)
    
    # Realice la imputación de los datos utilizando 
    # 3 aproximaciones diferentes y compare los resultados.

    realizar_imputacion(df)

    # Realice una estimación de valores 
    # faltantes mediante interpolación.

    realizar_interpolacion(df)