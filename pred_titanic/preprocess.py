import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import KNNImputer

class Preprocessor():
    '''only X data will be preprocessed with this class'''
    def __init__(self, df_train):
        self.df_train = df_train
        self.fit()

    def fit(self):
        '''fit data to set configs'''
        # group columns
        self.num_cols, self.cat_cols = self._group_colummns_by_types()
        self.onehot_cols = self._filter_onehot_target()

        # onehot encoding
        self.ohe = OneHotEncoder(handle_unknown='ignore')
        self.ohe = self.ohe.fit(self.df_train[self.onehot_cols])
        encoded_df = pd.DataFrame(
            self.ohe.transform(
            self.df_train[self.onehot_cols]).toarray(),
            columns=self.ohe.get_feature_names_out())

        df = pd.concat([self.df_train, encoded_df], axis=1)
        df = df.drop(columns=self.cat_cols)
        
        # numeric scaling
        self.scaler = StandardScaler()
        self.scaler = self.scaler.fit(df[self.num_cols])

        # nan imputate
        self.imputer = KNNImputer()
        self.imputer = self.imputer.fit(df)


    def transform(self, df):
        # onehot encoding 
        encoded_df = pd.DataFrame(
            self.ohe.transform(df[self.onehot_cols]).toarray(),
            columns=self.ohe.get_feature_names_out())

        df = pd.concat([df, encoded_df], axis=1)
        df.drop(columns=self.cat_cols, inplace=True)

        # numeric scaling
        df[self.num_cols] = self.scaler.transform(df[self.num_cols])

        # nan imputate
        df = pd.DataFrame(self.imputer.transform(df), columns=df.columns)

        return df

    def _group_colummns_by_types(self):
        num_cols = []
        cat_cols = []

        for colname in self.df_train.columns:
            if self.df_train[colname].dtype in ['int64', 'float64']:
                num_cols.append(colname)
            else:
                cat_cols.append(colname)

        return num_cols, cat_cols
    
    def _filter_onehot_target(self):
        '''전체 n수 대비 10% 이하의 클래스를 가지는 데이터는 onehot encoding'''
        onehot_cols = []
        for colname in self.cat_cols:
            if len(self.df_train[colname].unique()) \
                <= self.df_train.shape[0]*0.1:
                onehot_cols.append(colname)
        return onehot_cols
