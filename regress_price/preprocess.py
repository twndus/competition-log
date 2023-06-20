import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import KNNImputer

class Preprocessor():
    '''only X data will be preprocessed with this class'''
    def __init__(self, df_train, dataname):
        self.df_train = df_train
        self.dataname = dataname
        self.preprocess_config = {}
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
        
        if self.dataname =='titanic':
            df = self.preprocess_titanic(df, 'fit')
        elif self.dataname == 'house':
            df = self.preprocess_house(df, 'fit')
        
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
        
        if self.dataname =='titanic':
            df = self.preprocess_titanic(df, 'transform')
        elif self.dataname == 'house':
            df = self.preprocess_house(df, 'transform')

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

    def preprocess_titanic(self, df, step='transform'):#, num_cols, cat_cols):
        #Creating new family_size column
        df['Family_Size']=df['SibSp']+df['Parch']
        df['Age*Class']=df['Age']*df['Pclass']
        df['Fare_Per_Person']=df['Fare']/(df['Family_Size']+1)
        
        # custom log transformation
        for colname in ['SibSp', 'Parch', 'Fare']:
            df[['SibSp', 'Parch', 'Fare']] = \
                    df[['SibSp', 'Parch', 'Fare']].apply(lambda x: np.log1p(x))
        return df

    def preprocess_house(self, df, step='transform'):#, num_cols, cat_cols):
        # log transform
        df['WoodDeckSF'] = np.log1p(df['WoodDeckSF'])
        
        # add features
        df["TotalHouse"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]   
        df["TotalArea"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"] + df["GarageArea"]
        
        df["+_TotalHouse_OverallQual"] = df["TotalHouse"] * df["OverallQual"]
        df["+_GrLivArea_OverallQual"] = df["GrLivArea"] * df["OverallQual"]
        df["+_BsmtFinSF1_OverallQual"] = df["BsmtFinSF1"] * df["OverallQual"]
        
        df["-_LotArea_OverallQual"] = df["LotArea"] * df["OverallQual"]
        df["-_TotalHouse_LotArea"] = df["TotalHouse"] + df["LotArea"]
       
        df["Bsmt"] = df["BsmtFinSF1"] + df["BsmtFinSF2"] + df["BsmtUnfSF"]
        df["Rooms"] = df["FullBath"]+df["TotRmsAbvGrd"]
        df["PorchArea"] = df["OpenPorchSF"]+df["EnclosedPorch"]+df["3SsnPorch"]+df["ScreenPorch"]
        df["TotalPlace"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"] + \
            df["GarageArea"] + df["OpenPorchSF"]+df["EnclosedPorch"]+\
            df["3SsnPorch"]+df["ScreenPorch"]


        # outlier detection & treat
        if step == 'fit':
            self.num_cols += ['TotalHouse', 'TotalArea', '+_TotalHouse_OverallQual',
                    '+_GrLivArea_OverallQual', '+_BsmtFinSF1_OverallQual', '-_LotArea_OverallQual',
                    '-_TotalHouse_LotArea', 'Bsmt', 'Rooms', 'PorchArea', 'TotalPlace']
            for col in self.num_cols:
                iqr = df[col].quantile(q=.75) - df[col].quantile(q=.25)
                lower_bound = df[col].quantile(q=.25) - iqr * 1.5
                upper_bound = df[col].quantile(q=.75) + iqr * 1.5
                
                df.loc[df[col] < lower_bound, col] = lower_bound
                df.loc[df[col] > upper_bound, col] = upper_bound
                self.preprocess_config[col] = \
                    {'lower_bound': lower_bound, 'upper_bound': upper_bound}
        elif step == 'transform':
            for col in self.num_cols:
                lower_bound = self.preprocess_config[col]['lower_bound']
                upper_bound = self.preprocess_config[col]['upper_bound']
                
                df.loc[df[col] < lower_bound, col] = lower_bound
                df.loc[df[col] > upper_bound, col] = upper_bound

        return df
