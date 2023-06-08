import os, json, pickle

import pandas as pd
from sklearn.model_selection import StratifiedKFold

def data_loader(path, format='csv'):
    if format == 'csv':
        df = pd.read_csv(path)
    return df

def kfold_split(df_X, df_y, dataname, n_splits=5):
    skf_path = f'results/{dataname}-{n_splits}skf.pkl'

    def _load_skf_stored(skf_path):
        with open(skf_path, 'rb') as f:
            return pickle.load(f)
        
    def _store_folded_data(skf_path):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        skf_data = {}
        for i, (train_index, val_index) in enumerate(skf.split(df_X, df_y, df_y)):
            X_train, X_val = df_X.iloc[train_index,:], df_X.iloc[val_index,:]
            y_train, y_val = df_y.iloc[train_index,:], df_y.iloc[val_index,:]
            skf_data[f'{i}th'] = {
                'X_train': X_train, 'X_val': X_val, 
                'y_train': y_train, 'y_val': y_val
                }
        with open(skf_path, 'wb') as f:
            pickle.dump(skf_data, f, pickle.HIGHEST_PROTOCOL)

    if not os.path.exists(skf_path):
        _store_folded_data(skf_path)
    
    skf_data = _load_skf_stored(skf_path)
    return skf_data
    

        