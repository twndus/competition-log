import os

import numpy as np

from preprocess import Preprocessor, data_augmentation
from data import data_loader, kfold_split
from models import get_classifier, optimize, retrain, train_keras
from evaluate import evaluate
from submission import submission
from config import args, keras_params

from sklearn.metrics import accuracy_score

def main():#**args):
    # load train, test data
    train_df = data_loader(args['train_path'], format='csv')
    test_df = data_loader(args['test_path'], format='csv')

    # split X, y
    train_X = train_df.drop(columns=['id', 'EC1', 'EC2', 'EC3', 'EC4', 'EC5', 'EC6'])
    train_y = train_df[['EC1', 'EC2']]#.astype('int32')
    test_X = test_df.drop(columns=['id'])

    # preprocess fit, transform
    preprocessor = Preprocessor(train_X, args['dataname'])
    train_X = preprocessor.transform(train_X, train=True)
    test_X = preprocessor.transform(test_X)

    # oversampling
    if args['oversampling']:
        train_X, train_y = data_augmentation(train_X, train_y)

    # split data with CV
    data_splited = kfold_split(
        train_X, train_y, args['dataname'], n_splits=10)

#    if args['modelname'].endswith('_keras'):
#        # retrain and get preds
#        test_preds, train_metrics = train_keras(
#            args['modelname'],
#            keras_params,
#            args['task'],
#            data_splited,
#            test_X
#        )
#
#        print("test_preds: ",test_preds)
#    else:
    test_preds, train_metrics = [], []
    for modelname in args['modelname'].split('+'):
        if modelname.endswith('_keras'):
            # retrain and get preds
            pred_array, train_metric = train_keras(
                modelname,
                keras_params,
                args['task'],
                data_splited,
                test_X
            )

        else:
            # get best parameters with optuna
            best_params = optimize(
                modelname,
                args['task'],
                train_X, train_y
            )
            # retrain and get preds
            pred_array, train_metric = retrain(
                modelname,
                best_params,
                args['task'],
                data_splited,
                test_X
            )

        if args['task'] == 'classification':
            test_pred = np.mean(pred_array, axis=0)
            train_metric = np.mean(train_metric, axis=0)
            
        test_preds.append(test_pred)
        train_metrics.append(train_metric)

    test_pred = np.mean(test_preds, axis=0)
    train_metric = np.mean(train_metrics, axis=0)

    # submission
    submission_df = data_loader(args['submission_path'], format='csv')
    submission(test_pred, submission_df, args['modelname'],
        train_metrics=f'{train_metric:.4f}')

#    else:
#        # get best parameters with optuna
#        best_params = optimize(
#            args['modelname'],
#            args['task'],
#            train_X, train_y
#        )
#        
#    #    best_params = {'learning_rate': 0.00001}
#        # retrain and get preds
#        pred_array, train_metrics = retrain(
#            args['modelname'],
#            best_params,
#            args['task'],
#            data_splited,
#            test_X
#        )

#    # voting
#    if args['task'] == 'classification':
#        test_pred = np.mean(pred_array, axis=0)
#        train_metric = np.mean(train_metrics, axis=0)
##        test_pred = np.apply_along_axis(
##                lambda x: np.argmax(np.bincount(x)), axis=0, arr=pred_array)
##        train_metric = np.mean(train_metrics)
##    elif args['task'] == 'regression':
##        test_pred = np.mean(pred_array, axis=0)
##        train_metric = np.mean(train_metrics, axis=0)
#
#    # submission
#    submission_df = data_loader(args['submission_path'], format='csv')
#    submission(test_pred, submission_df, args['modelname'],
#        train_metrics=f'{train_metric:.4f}')

if __name__ == '__main__':
    main()
