'''
Classification
- (+) logistic regression
- (+) knn (ml)
- (+) svc
- (+) adaboost 
- (+) random forest 
- (+) mlp
- (+) gbm
- (+) xgboost

Regression
- (+) knn (ml)
- ( ) linear regression
- ( ) lasso
- ( ) elasticnet
- (+) mlp
- (+) random forest 
- (+) catboost
- (+) extra trees 
'''
from functools import partial

import numpy as np

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor 
from sklearn.ensemble import (
        RandomForestClassifier, AdaBoostClassifier, AdaBoostRegressor,
        RandomForestRegressor)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier, HistGradientBoostingRegressor,
    ExtraTreesClassifier)
from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.multioutput import MultiOutputClassifier

from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
import optuna
import tensorflow as tf

from evaluate import evaluate
from keras_models import mlpclassifier_keras
import config
from model_params import classification_params, regression_params

def get_params(modelname, params_, trial):
    params = {}
    for p, v in params_[modelname].items():
        if v['type'] == 'float':
            if v['values'][3]:
                params[p] = trial.suggest_float(p,
                    v['values'][0], v['values'][1], log=v['values'][3])
            else:
                params[p] = trial.suggest_float(p,
                    v['values'][0], v['values'][1], step=v['values'][2])
        elif v['type'] == 'cat' or (v['type'] == 'int' and v['continuous'] == 'False'):
            params[p] = trial.suggest_categorical(p, v['values'])
        elif v['type'] == 'int':
            if v['values'][3]:
                params[p] = trial.suggest_int(p,
                    v['values'][0], v['values'][1], log=v['values'][3])
            else:
                params[p] = trial.suggest_int(p,
                    v['values'][0], v['values'][1], step=v['values'][2])
    return params

def get_model(modelname='knn', task=None, input_shape=None, **args):
    if task == 'classification':
        model = get_classifier(modelname, input_shape=input_shape, **args)
    elif task == 'regression':
        model = get_regressor(modelname, input_shape=input_shape, **args)
    if config.args['multi-label']:
        model = MultiOutputClassifier(model, n_jobs=-1)
    return model

def get_regressor(modelname='knn', input_shape=None, **args):
    if modelname == 'knn':
        model = KNeighborsRegressor(**args)
    elif modelname == 'rf':
        model = RandomForestRegressor(**args)
    elif modelname == 'mlp':
        model = MLPRegressor(warm_start=True, **args)
    elif modelname == 'gbm':
        model = HistGradientBoostingRegressor(warm_start=True, **args)
    elif modelname == 'ada':
        model = AdaBoostRegressor(DecisionTreeRegressor(), **args)
    return None

def get_classifier(modelname='knn', input_shape=None, **args):
    if modelname == 'knn':
        model = KNeighborsClassifier(**args)
    elif modelname == 'logistic':
        model = LogisticRegression(**args)
    elif modelname == 'svc':
        model = SVC(**args)
    elif modelname == 'rf':
        model = RandomForestClassifier(warm_start=True, **args)
    elif modelname == 'ada':
        model = AdaBoostClassifier(**args)
    elif modelname == 'mlp':
        model = MLPClassifier(warm_start=True, **args)
    elif modelname == 'gbm':
        model = HistGradientBoostingClassifier(warm_start=True, **args)
    elif modelname == 'xgboost':
        model = XGBClassifier(**args)
    elif modelname == 'catboost':
        model = CatBoostClassifier(**args)
    elif modelname == 'extratree':
        model = ExtraTreesClassifier(warm_start=True, **args)
    elif modelname == 'mlp_keras':
        learning_rate = args['learning_rate']
        model = mlpclassifier_keras(input_shape, learning_rate)
    elif modelname == 'lgbm':
        model = LGBMClassifier(**args)
    return model

def classification_objective(trial, modelname, train_X, train_y):
    params = get_params(modelname, classification_params, trial)
    model = get_model(
        modelname, task='classification', input_shape=train_X.shape[1:], **params)
    if config.args['metric'] == 'auc':
#            score = cross_val_score(
#                model, train_X, train_y, scoring="roc_auc",
#                n_jobs=-1, cv=3)
            model.fit(train_X, train_y)
            y_pred = model.predict(train_X)
            score = roc_auc_score(train_y, y_pred, multi_class='ovr')
    else:
        score = cross_val_score(
            model, train_X, train_y, n_jobs=-1, cv=3)
        score = score.mean()
    return score

def regression_objective(trial, modelname, train_X, train_y):
    params = get_params(modelname, regression_params, trial)
    model = get_model(
        modelname, task='regression', input_shape=train_X.shape[1:], **params)
    
    score = cross_val_score(
            model, train_X, train_y, scoring='neg_mean_squared_error', n_jobs=-1, cv=3)
    rmse = score.mean()
    return rmse

def retrain(modelname, best_params, task, data_splited, test_X):
    pred_list = []
    train_accs = []
    
    if 'classifier' in best_params.keys():
        del best_params['classifier']
    
    for i in range(len(data_splited.keys())):
        if modelname.endswith('keras'):

            # learning params
            epochs = 500
            batch_size = 20

            # model
            model = get_classifier(modelname, task, input_shape=test_X.shape[1:], **best_params)
            
            # callback
            callback = tf.keras.callbacks.EarlyStopping(patience=5)

            history = model.fit(
                data_splited[f'{i}th']['X_train'], 
                data_splited[f'{i}th']['y_train'], epochs=epochs, 
                validation_data=(data_splited[f'{i}th']['X_val'], 
                data_splited[f'{i}th']['y_val']),
                    batch_size=batch_size, callbacks=[callback])
            train_eval = model.evaluate(data_splited[f'{i}th']['X_train'], 
                data_splited[f'{i}th']['y_train'])[1]
            test_pred = (model.predict(test_X) > 0.5).astype(int)
        else:
            # define model
            model = get_model(
                modelname, task='classification', input_shape=None, 
                **best_params)
            
            # train
            model = model.fit(data_splited[f'{i}th']['X_train'].values, 
                data_splited[f'{i}th']['y_train'].values)
        
            # evaluate
            train_pred = model.predict(data_splited[f'{i}th']['X_train'].values)
            val_pred = model.predict(data_splited[f'{i}th']['X_val'].values)
            
            train_eval = evaluate(data_splited[f'{i}th']['y_train'], train_pred, 
                     desc='train')
            _ = evaluate(data_splited[f'{i}th']['y_val'], val_pred, desc='val')
            print(train_eval)
            test_pred = model.predict(test_X)
        
        # pred
        pred_list.append(test_pred)
        train_accs.append(train_eval)

    return np.array(pred_list), train_accs

def retrain_whole(modelname, best_params, task, train_X, train_y, test_X):
    pred_list = []
    train_accs = []
    
    if 'classifier' in best_params.keys():
        del best_params['classifier']
    
    # define model
    model = get_model(
        modelname, task='classification', input_shape=None, 
        **best_params)
    
    # train
    model = model.fit(train_X, train_y)

    # evaluate
    train_pred = model.predict(train_X)    
    train_eval = evaluate(train_y, train_pred, desc='train')
    print(train_eval)
    
    test_pred = model.predict(test_X)
    
    # pred
    pred_list.append(test_pred)
    train_accs.append(train_eval)

    return np.array(pred_list), train_accs


def optimize(modelname, task, train_X, train_y):
    study = optuna.create_study(direction='maximize')
    if task == 'classification':
        objective = partial(
            classification_objective, modelname=modelname, 
            train_X=train_X.values, train_y=train_y.values
        )
    elif task == 'regression':
        objective = partial(
            regression_objective, modelname=modelname, 
            train_X=train_X.values, train_y=train_y.values
        )
    study.optimize(objective, n_trials=30)
    return study.best_params
