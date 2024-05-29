import pandas as pd

def intervalos(lugar, mes):
    filename = 'intervalos.csv'
    df = pd.read_csv(filename)
    
    # Filtrar los datos según las condiciones especificadas
    filtro_Call_Hdd = (df['Region'] == lugar) & (df['Month'] == mes) & (df['Variable'] == 'Call_HDD')
    Call_Hdd = df.loc[filtro_Call_Hdd, 'Confidence Interval'].values

    filtro_Put_HDD = (df['Region'] == lugar) & (df['Month'] == mes) & (df['Variable'] == 'Put_HDD')
    Put_HDD = df.loc[filtro_Put_HDD, 'Confidence Interval'].values

    filtro_Call_CDD = (df['Region'] == lugar) & (df['Month'] == mes) & (df['Variable'] == 'Call_CDD')
    Call_CDD = df.loc[filtro_Call_CDD, 'Confidence Interval'].values

    filtro_Put_CDD = (df['Region'] == lugar) & (df['Month'] == mes) & (df['Variable'] == 'Put_CDD')
    Put_CDD = df.loc[filtro_Put_CDD, 'Confidence Interval'].values

    # Convertir los resultados a strings, si están disponibles
    Call_Hdd_str = str(Call_Hdd[0]) if len(Call_Hdd) > 0 else None
    Put_HDD_str = str(Put_HDD[0]) if len(Put_HDD) > 0 else None
    Call_CDD_str = str(Call_CDD[0]) if len(Call_CDD) > 0 else None
    Put_CDD_str = str(Put_CDD[0]) if len(Put_CDD) > 0 else None

    return Call_Hdd_str, Put_HDD_str, Call_CDD_str, Put_CDD_str

# Ejemplo de uso:
print(intervalos("Anti", 2))
