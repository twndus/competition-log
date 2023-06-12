import os

import numpy as np

from preprocess import Preprocessor
from data import data_loader, kfold_split
from models import get_classifier, optimize, retrain, mlpclassifier_keras
from evaluate import evaluate
from submission import submission

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
  
#    if kwargs['modelname'].endswith('keras'):
##        learning_rate = 0.0001
##        model = mlpclassifier_keras(train_X.shape[1:], learning_rate)
##        history = model.fit(train_X, train_y, epochs=200, validation_split=0.2)
#        train_eval = model.evaluate(train_X, train_y)
#        print(f'{train_eval[-1]:.4f}')
#        test_pred = (model.predict(test_X) > 0.5).astype(int)
#        
#        # submission
#        submission_df = data_loader(kwargs['submission_path'], format='csv')
#        submission(test_pred, submission_df, kwargs['modelname'], 
#            train_acc=f'{train_eval[-1]:.4f}')
#    else:
    # get best parameters with optuna
    best_params = optimize(
        kwargs['modelname'],
        train_X, train_y
    )

    # retrain and get preds
    pred_array = retrain(
        kwargs['modelname'],
        best_params,
        data_splited,
        test_X
    )

    # voting
    test_pred = np.apply_along_axis(
            lambda x: np.argmax(np.bincount(x)), axis=0, arr=pred_array)

    # submission
    submission_df = data_loader(kwargs['submission_path'], format='csv')
    submission(test_pred, submission_df, kwargs['modelname'])


if __name__ == '__main__':
    # args
    data_dir = '../../../datasets/kaggle/titanic'
    train_path = os.path.join(data_dir, 'train.csv')
    test_path = os.path.join(data_dir, 'test.csv')
    submission_path = os.path.join(data_dir, 'gender_submission.csv')
    modelname = 'mlp_keras' #'gbm'#'mlp' #'rf' #'svc' #'logistic' #'knn'
    dataname = 'titanic'

    main(
        train_path=train_path, test_path=test_path, 
        submission_path=submission_path, modelname=modelname, 
        dataname=dataname,
    )
