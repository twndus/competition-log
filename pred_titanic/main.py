import os

import numpy as np

from submission import submission
from preprocess import Preprocessor
from data import data_loader, kfold_split
from models import get_classifier

def main(**kwargs):
    # load train, test data
    train_df = data_loader(kwargs['train_path'], format='csv')
    test_df = data_loader(kwargs['test_path'], format='csv')

    # split X, y
    train_X = train_df.drop(columns=['PassengerId', 'Survived'])
    train_y = train_df[['Survived']]
    test_X = test_df.drop(columns=['PassengerId'])
    
    # print("train_X.columns: ", train_X.columns)
    # print("train_y.columns: ", train_y.columns)
    # print("test_X.columns: ", test_X.columns)

    # preprocess fit, transform
    preprocessor = Preprocessor(train_X)
    train_X = preprocessor.transform(train_X)
    test_X = preprocessor.transform(test_X)

    # print("train_X.columns: ", train_X.columns)
    # print("test_X.columns: ", test_X.columns)

    # split data with CV
    data_splited = kfold_split(
        train_X, train_y, kwargs['dataname'], n_splits=5)
    
    # model
    model = get_classifier('knn')
    print(model)

    # # train
    # model = model.fit(
    #     data_splited['1th']['train_X'], 
    #     data_splited['1th']['train_y']
    #     )
    # evaluate
    # predict
    
    # submission
    submission_df = data_loader(kwargs['submission_path'], format='csv')
    pred_array = np.random.normal(0, 1, 418)
    submission(pred_array, submission_df, modelname)

if __name__ == '__main__':
    data_dir = '../../../datasets/kaggle/titanic'
    train_path = os.path.join(data_dir, 'train.csv')
    test_path = os.path.join(data_dir, 'test.csv')
    submission_path = os.path.join(data_dir, 'gender_submission.csv')
    modelname = 'none'
    dataname = 'titanic'

    main(
        train_path=train_path, test_path=test_path, 
        submission_path=submission_path, modelname=modelname, 
        dataname=dataname,
    )