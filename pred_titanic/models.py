'''
Classification
- ( ) logistic regression
- (+) knn (ml)
- ( ) svc
- ( ) decision tree
- ( ) mlp

Regression
- ( ) linear regression
- ( ) lasso
- ( ) elasticnet
- ( ) mlp
'''
from functools import partial

import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import optuna

from evaluate import evaluate

## params
params = {
    'knn': {
        'n_neghbors': 5,
    }
}

def get_classifier(name='knn', **args):
    if name == 'knn':
        if len(args.keys()):
            model = KNeighborsClassifier(**args)
        else:
            model = KNeighborsClassifier()
    return model


def classification_objective(trial, modelname, train_X, train_y):
    classifier_name = trial.suggest_categorical('classifier', ['knn'])
    if modelname == 'knn':
        knn_n = trial.suggest_int('n_neighbors', 3, 7)
        classifier_obj = KNeighborsClassifier()
    
    classifier_obj = KNeighborsClassifier()
    
    score = cross_val_score(
            classifier_obj, train_X, train_y, n_jobs=-1, cv=3)
    accuracy = score.mean()
    return accuracy

def retrain(modelname, best_params, data_splited, test_X):
    pred_list = []
    del best_params['classifier']
    
    for i in range(len(data_splited.keys())):
        # model
        model = get_classifier(modelname, **best_params)
        # train
        model = model.fit(data_splited[f'{i}th']['X_train'], 
            data_splited[f'{i}th']['y_train'])
    
        # evaluate
        train_pred = model.predict(data_splited[f'{i}th']['X_train'])
        val_pred = model.predict(data_splited[f'{i}th']['X_val'])
    
        evaluate(data_splited[f'{i}th']['y_train'], train_pred, 
                 metric='accuracy', desc='train')
        evaluate(data_splited[f'{i}th']['y_val'], val_pred, 
                 metric='accuracy', desc='val')
        
        # pred
        pred_list.append(model.predict(test_X))

    return np.array(pred_list)

def optimize(modelname, train_X, train_y):
    study = optuna.create_study(direction='maximize')
    objective = partial(
        classification_objective, modelname=modelname, 
        train_X=train_X, train_y=train_y
    )
    study.optimize(objective, n_trials=100)
    return study.best_params
