import os

import numpy as np

from submission import submission
from preprocess import Preprocessor
from data import data_loader, kfold_split
from models import get_classifier
from evaluate import evaluate

from sklearn.metrics import accuracy_score

def main(**kwargs):
    # load train, test data
    train_df = data_loader(kwargs['train_path'], format='csv')
    test_df = data_loader(kwargs['test_path'], format='csv')

    # split X, y
    train_X = train_df.drop(columns=['PassengerId', 'Survived'])
    train_y = train_df[['Survived']]
    test_X = test_df.drop(columns=['PassengerId'])

    # preprocess fit, transform
    preprocessor = Preprocessor(train_X)
    train_X = preprocessor.transform(train_X)
    test_X = preprocessor.transform(test_X)

    # split data with CV
    data_splited = kfold_split(
        train_X, train_y, kwargs['dataname'], n_splits=5)
    
    # model
    model = get_classifier(kwargs['modelname'])
    print(model)

    # train
    model = model.fit(
        data_splited['1th']['X_train'], 
        data_splited['1th']['y_train']
        )
    
    # evaluate

    train_pred = model.predict(data_splited['1th']['X_train'])
    val_pred = model.predict(data_splited['1th']['X_val'])
    
    evaluate(data_splited['1th']['y_train'], train_pred, 
             metric='accuracy', desc='train')
    evaluate(data_splited['1th']['y_val'], val_pred, 
             metric='accuracy', desc='val')
    
    # pred
    test_pred = model.predict(test_X)
    
    # submission
    submission_df = data_loader(kwargs['submission_path'], format='csv')
    submission(test_pred, submission_df, kwargs['modelname'])


if __name__ == '__main__':
    # args
    data_dir = '../../../datasets/kaggle/titanic'
    train_path = os.path.join(data_dir, 'train.csv')
    test_path = os.path.join(data_dir, 'test.csv')
    submission_path = os.path.join(data_dir, 'gender_submission.csv')
    modelname = 'knn'
    dataname = 'titanic'

    main(
        train_path=train_path, test_path=test_path, 
        submission_path=submission_path, modelname=modelname, 
        dataname=dataname,
    )