import pandas as pd

def data_loader(path, format='csv'):
    if format == 'csv':
        df = pd.read_csv(path)
    return df