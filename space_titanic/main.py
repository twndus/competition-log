import os

import numpy as np

from preprocess import Preprocessor
from data import data_loader, kfold_split
from models import get_classifier, optimize, retrain
from evaluate import evaluate
from submission import submission

from sklearn.metrics import accuracy_score

def main(**kwargs):
    # load train, test data
    train_df = data_loader(kwargs['train_path'], format='csv')
    test_df = data_loader(kwargs['test_path'], format='csv')

    # split X, y
    train_X = train_df.drop(columns=['PassengerId', 'Transported'])
    train_y = train_df[['Transported']]
    test_X = test_df.drop(columns=['PassengerId'])

    # preprocess fit, transform
    preprocessor = Preprocessor(train_X, kwargs['dataname'])
    train_X = preprocessor.transform(train_X)
    test_X = preprocessor.transform(test_X)

    # split data with CV
    data_splited = kfold_split(
        train_X, train_y, kwargs['dataname'], n_splits=5)
  
    # get best parameters with optuna
    best_params = optimize(
        kwargs['modelname'],
        kwargs['task'],
        train_X, train_y
    )
    
    print(best_params)

#    best_params = {'learning_rate': 0.001}
    # retrain and get preds
    pred_array, train_metrics = retrain(
        kwargs['modelname'],
        best_params,
        kwargs['task'],
        data_splited,
        test_X
    )

    # voting
    if task == 'classification':
        test_pred = np.apply_along_axis(
                lambda x: np.argmax(np.bincount(x)), axis=0, arr=pred_array)
        train_metric = np.mean(train_metrics)
    elif task == 'regression':
        test_pred = np.mean(pred_array, axis=0)
        train_metric = np.mean(train_metrics, axis=0)

    # submission
    submission_df = data_loader(kwargs['submission_path'], format='csv')
    submission(test_pred, submission_df, kwargs['modelname'],
        train_metrics=f'{train_metric:.4f}', sub_type='bool')


if __name__ == '__main__':
    # args
    data_dir = '/home/jylee/datasets/kaggle/spaceship-titanic'
    train_path = os.path.join(data_dir, 'train.csv')
    test_path = os.path.join(data_dir, 'test.csv')
    submission_path = os.path.join(data_dir, 'sample_submission.csv')
    task = 'classification'
    modelname = 'xgboost'#'mlp_keras' #'gbm'#'mlp' #'rf' #'svc' #'logistic' #'knn'
    dataname = 'spaceship'
    epochs = 20

    main(
        train_path=train_path, test_path=test_path, 
        submission_path=submission_path, modelname=modelname, 
        task=task, dataname=dataname, epochs=epochs,
    )
