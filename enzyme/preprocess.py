import numpy as np
import pandas as pd

from scipy.stats import skew
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import KNNImputer

from skmultilearn.problem_transform import LabelPowerset
from imblearn.over_sampling import RandomOverSampler

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
        self.num_cols, self.bin_cols, self.cat_cols = self._group_columns_by_types()
        self.onehot_cols = self._filter_onehot_target()
        
        df = self.df_train

        # custom preprocess
        if self.dataname =='titanic':
            df = self.preprocess_titanic(df, 'fit')
        elif self.dataname == 'house':
            df = self.preprocess_house(df, 'fit')
        elif self.dataname == 'spaceship':
            df = self.preprocess_spaceship(df, 'fit')
        elif self.dataname == 'machine':
            df = self.preprocess_machine(df, 'fit')
        elif self.dataname == 'enzyme':
            df = self.preprocess_enzyme(df, 'fit')

        # categorical data imputation
        df = self._imputate_cat_cols(df)

        # onehot encoding
        self.ohe = OneHotEncoder(handle_unknown='ignore')
        self.ohe = self.ohe.fit(df[self.onehot_cols])
        encoded_df = pd.DataFrame(
            self.ohe.transform(
            df[self.onehot_cols]).toarray(),
            columns=self.ohe.get_feature_names_out())

        df = pd.concat([df, encoded_df], axis=1)
        df = df.drop(columns=self.cat_cols)
        
        # skewed data
        self.pos_skewed, self.neg_skewed = self._filter_skewed_cols(df)
        df = self._log_transform(df)
        df = self._exp_transform(df)

        # if self.dataname == 'enzyme':
        #     df = self.preprocess_enzyme(df, 'fit')

        # numeric scaling
        self.scaler = StandardScaler()
        self.scaler = self.scaler.fit(df[self.num_cols])

        # nan imputate
        self.imputer = KNNImputer()
        self.imputer = self.imputer.fit(df[self.num_cols])

        # get mode

    def transform(self, df, train=False):
        # custom preprocess
        if self.dataname =='titanic':
            df = self.preprocess_titanic(df, 'transform')
        elif self.dataname == 'house':
            df = self.preprocess_house(df, 'transform')
        elif self.dataname == 'spaceship':
            df = self.preprocess_spaceship(df, 'transform')
        elif self.dataname == 'machine':
            df = self.preprocess_machine(df, 'transform')
        elif self.dataname == 'enzyme':
            df = self.preprocess_enzyme(df, 'transform')
    
        # categorical data imputation
        df = self._imputate_cat_cols(df, 'transform')

        # onehot encoding 
        encoded_df = pd.DataFrame(
            self.ohe.transform(df[self.onehot_cols]).toarray(),
            columns=self.ohe.get_feature_names_out())

        df = pd.concat([df, encoded_df], axis=1)
        df.drop(columns=self.cat_cols, inplace=True)

        # skewed data
        df = self._log_transform(df)
        df = self._exp_transform(df)

        # if self.dataname == 'enzyme':
        #     df = self.preprocess_enzyme(df, 'transform')

        # numeric scaling
        df[self.num_cols] = self.scaler.transform(df[self.num_cols])

        # nan imputate
        df[self.num_cols] = self.imputer.transform(df[self.num_cols])

        return df

    def _group_columns_by_types(self):
        num_cols = []
        bin_cols = []
        cat_cols = []

        for colname in self.df_train.columns:
            if self.df_train[colname].dtype in ['int64', 'float64']:
                if len(self.df_train[colname].dropna().unique()) == 2:
                    bin_cols.append(colname)
                else:
                    num_cols.append(colname)
            else:
                cat_cols.append(colname)

        return num_cols, bin_cols, cat_cols
    
    def _filter_onehot_target(self):
        '''전체 n수 대비 10% 이하의 클래스를 가지는 데이터는 onehot encoding'''
        onehot_cols = []
        for colname in self.cat_cols:
            if len(self.df_train[colname].unique()) \
                <= self.df_train.shape[0]*0.1:
                onehot_cols.append(colname)
        return onehot_cols

    def _filter_skewed_cols(self, df):
        pos_skewed, neg_skewed = [], []

        for col in self.num_cols:
            # Calculate skewness
            skewness = skew(df[col], nan_policy='omit')
            exceptions = len(df[col].unique()) > 2
            if (skewness >= 1) and exceptions:
                pos_skewed.append(col)
            elif (skewness <= -1) and exceptions:
                neg_skewed.append(col)

        return pos_skewed, neg_skewed
    
    def _log_transform(self, df):
        for col in self.pos_skewed:
            df[col] = np.log1p(df[col])
        return df

    def _exp_transform(self, df):
        for col in self.neg_skewed:
            df[col] = np.expm1(df[col])
        return df
    
    def _imputate_cat_cols(self, df, step='fit'):
        if step == 'fit':
            for col in self.onehot_cols:
                mode = df[col].mode()[0]
                df[col].fillna(mode, inplace=True)

                if col not in self.preprocess_config.keys():
                    self.preprocess_config[col] = {}
                self.preprocess_config[col]['mode'] = mode
        else:
            for col in self.onehot_cols:
                df[col].fillna(self.preprocess_config[col]['mode'], inplace=True)

        return df

    def preprocess_titanic(self, df, step='transform'):
        #Creating new family_size column
        df['Family_Size']=df['SibSp']+df['Parch']
        df['Age*Class']=df['Age']*df['Pclass']
        df['Fare_Per_Person']=df['Fare']/(df['Family_Size']+1)
        
        # custom log transformation
        for colname in ['SibSp', 'Parch', 'Fare']:
            df[['SibSp', 'Parch', 'Fare']] = \
                    df[['SibSp', 'Parch', 'Fare']].apply(lambda x: np.log1p(x))
        return df

    def preprocess_house(self, df, step='transform'):
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

    def preprocess_spaceship(self, df, step='transform'):
        # cabin split
        df['deck'] = df['Cabin'].apply(lambda x: x.split('/')[0] if type(x) != float else np.nan)
        df['num'] = df['Cabin'].apply(lambda x: x.split('/')[1] if type(x) != float else np.nan).astype('float')
        df['side'] = df['Cabin'].apply(lambda x: x.split('/')[2] if type(x) != float else np.nan)
        df['total_amount'] = df[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].sum(axis=1)

        df['GroupId'] = df['PassengerId'].apply(lambda x: x.split('_')[0])
        df['GroupSize'] = df.groupby("GroupId")["GroupId"].transform("count")
        df["Alone"] = df["GroupSize"] == 1

        if step == 'fit':
            self.num_cols.extend(['num', 'total_amount', 'GroupSize'])
            self.cat_cols.extend(['deck', 'side', 'Alone'])
        elif step == 'transform':
            # pass
            df.drop(columns=['GroupId'], inplace=True)
        return df
    
    def preprocess_machine(self, df, step='transform'):
        df['Product Type'] = df['Product ID'].apply(lambda x: x[0])
        df['total'] = df[self.bin_cols].sum(axis=1)

        # Create a new feature by divided 'Air temperature' from 'Process temperature'
        df["Temperature ratio"] = df['Process temperature [K]'] / df['Air temperature [K]']
        
        # Create a new feature by multiplying 'Torque' and 'Rotational speed'
        df['Torque * Rotational speed'] = df['Torque [Nm]'] * df['Rotational speed [rpm]']

        # Create a new feature by multiplying 'Torque' by 'Tool wear'
        df['Torque * Tool wear'] = df['Torque [Nm]'] * df['Tool wear [min]']
        
        # Create a new feature by multiplying 'Torque' by 'Rotational speed'
        df['Torque * Rotational speed'] = df['Torque [Nm]'] * df['Rotational speed [rpm]']
        
        if step == 'fit':
            self.cat_cols.extend(['Product Type'])
            self.num_cols.extend(['total', 'Temperature ratio', 'Torque * Rotational speed', 
                'Torque * Tool wear', 'Torque * Rotational speed'])
            self.onehot_cols.remove('Product ID')
        elif step == 'transform':
            pass  
        return df

    def preprocess_enzyme(self, df, step='transform'):

        df['Chi_sum'] = df[['Chi1', 'Chi1n', 'Chi1v', 'Chi2n', 'Chi2v', 'Chi3v', 'Chi4n']].sum(axis=1)
        df['EState_sum'] = df[['EState_VSA1', 'EState_VSA2']].sum(axis=1)

        df['BertzCT_MaxAbsEStateIndex_Ratio'] = df['BertzCT'] / (df['MaxAbsEStateIndex'] + 1e-12)
        df['BertzCT_ExactMolWt_Product'] = df['BertzCT'] * df['ExactMolWt']
        df['NumHeteroatoms_FpDensityMorgan1_Ratio'] = df['NumHeteroatoms'] / (df['FpDensityMorgan1'] + 1e-12)
        df['VSA_EState9_EState_VSA1_Ratio'] = df['VSA_EState9'] / (df['EState_VSA1'] + 1e-12)
        df['PEOE_VSA10_SMR_VSA5_Ratio'] = df['PEOE_VSA10'] / (df['SMR_VSA5'] + 1e-12)
        df['Chi1v_ExactMolWt_Product'] = df['Chi1v'] * df['ExactMolWt']
        df['Chi2v_ExactMolWt_Product'] = df['Chi2v'] * df['ExactMolWt']
        df['Chi3v_ExactMolWt_Product'] = df['Chi3v'] * df['ExactMolWt']
        df['EState_VSA1_NumHeteroatoms_Product'] = df['EState_VSA1'] * df['NumHeteroatoms']
        df['PEOE_VSA10_Chi1_Ratio'] = df['PEOE_VSA10'] / (df['Chi1'] + 1e-12)
        df['MaxAbsEStateIndex_NumHeteroatoms_Ratio'] = df['MaxAbsEStateIndex'] / (df['NumHeteroatoms'] + 1e-12)
        df['BertzCT_Chi1_Ratio'] = df['BertzCT'] / (df['Chi1'] + 1e-12)
        df['Ring_Density'] = divide_with_check(df['NumHeteroatoms'], df['SlogP_VSA3'])
        df['Hydrophilic_Surface'] = df['VSA_EState9'] * df['EState_VSA2']
        
        if step == 'fit':
           self.num_cols.extend([
                'Chi_sum', 'EState_sum', 'BertzCT_MaxAbsEStateIndex_Ratio', 'BertzCT_ExactMolWt_Product', 
                'NumHeteroatoms_FpDensityMorgan1_Ratio', 'VSA_EState9_EState_VSA1_Ratio',
                'PEOE_VSA10_SMR_VSA5_Ratio', 'Chi1v_ExactMolWt_Product', 'Chi2v_ExactMolWt_Product', 
                'Chi3v_ExactMolWt_Product', 'EState_VSA1_NumHeteroatoms_Product', 'PEOE_VSA10_Chi1_Ratio',
                'MaxAbsEStateIndex_NumHeteroatoms_Ratio', 'BertzCT_Chi1_Ratio',
                'Ring_Density', 'Hydrophilic_Surface'])
        else:
            pass
        return df

def divide_with_check(a,b):
    result = np.where(b != 0, np.divide(a, b), 0)
    return result


def data_augmentation(train_X, train_y):
    lp = LabelPowerset()
    ros = RandomOverSampler(random_state=42)

    # Applies the above stated multi-label (ML) to multi-class (MC) transformation.
    yt = lp.transform(train_y)
    X_resampled, y_resampled = ros.fit_resample(train_X, yt)

    # Inverts the ML-MC transformation to recreate the ML set
    y_resampled = pd.DataFrame(lp.inverse_transform(y_resampled).toarray(), columns=train_y.columns)

    return X_resampled, y_resampled