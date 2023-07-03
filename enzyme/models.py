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
    ExtraTreesClassifier,
    )
from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from sklearn.model_selection import cross_val_score
import optuna
import tensorflow as tf

from evaluate import evaluate
from keras_models import mlpclassifier_keras
import config
from model_params import classification_params, regression_params

metrics = {'classification': 'accuracy',
        'regression': 'rmse'}

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
    return model

def classification_objective(trial, modelname, train_X, train_y):
    params = get_params(modelname, classification_params, trial)
    model = get_model(
        modelname, task='classification', input_shape=train_X.shape[1:], **params)
    if config.args['metric'] == 'auc':
            score = cross_val_score(
                model, train_X, train_y, scoring="roc_auc",
                n_jobs=-1, cv=3)
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
            # model
            model = get_classifier(modelname, task, **best_params)
            
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
    study.optimize(objective, n_trials=10)
    return study.best_params


# def classification_objective(trial, modelname, train_X, train_y):
#     classifier_name = trial.suggest_categorical('classifier', [modelname])
#     # params = classification_params[modelname]
#     params = get_params(modelname, trial)
#     if modelname == 'knn':
#         # arg_n = trial.suggest_categorical('n_neighbors', [3, 5, 7])
#         model = KNeighborsClassifier(**params)
#     elif modelname == 'logistic':
#         # arg_p = trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet'])
#         # arg_tol = trial.suggest_float('tol', 1e-5, 1e-2)
#         model = LogisticRegression(**params)
#     elif modelname == 'svc':
#         # arg_c = trial.suggest_float('C', 0.1, 1000, log=True)
#         # arg_gamma = trial.suggest_float('gamma', 0.0001, 1, log=True)
#         model = SVC(**params)
#     elif modelname == 'rf':
#         # params = {}
#         # params['criterion'] = trial.suggest_categorical('criterion', ['gini', 'entropy'])
#         # params['max_depth'] = trial.suggest_int('max_depth', 80, 110, step=10)
#         # params['max_features'] = trial.suggest_int('max_features', 2, 4)
#         # params['n_estimators'] = trial.suggest_int('n_estimators', 50, 500, step=100)
#         model = RandomForestClassifier(warm_start=True, **params)
#     elif modelname == 'ada':
#         # arg_estimators = trial.suggest_int('n_estimators', 100, 1000, step=100)
#         # arg_lr = trial.suggest_float('learning_rate', 0.0001, 1, log=True)
#         # arg_algorithms = trial.suggest_categorical('algorithm', ['SAMME', 'SAMME.R'])
#         model = AdaBoostClassifier(**params)
#     elif modelname == 'mlp':
#         # params = {}
#         # params['learning_rate_init'] = trial.suggest_float('learning_rate_init', 0.0001, 1, log=True)
#         # params['learning_rate'] = trial.suggest_categorical('learning_rate', ['constant', 'invscaling', 'adaptive'])
#         # params['alpha'] = trial.suggest_float('alpha', 0.0001, 1, log=True)
#         # params['activation'] = trial.suggest_categorical('activation', ['logistic', 'relu', 'tanh'])
#         # params['solver'] = trial.suggest_categorical('solver', ['lbfgs', 'sgd', 'adam'])
#         # params['batch_size'] = trial.suggest_int('batch_size', 1, 1000, log=True)
#         # params['hidden_layer_sizes'] = trial.suggest_int('hidden_layer_sizes', 1, 1000)
#         # params['max_iter'] = trial.suggest_int('max_iter', 100, 1000)
#         model = MLPClassifier(warm_start=True, **params)
#     elif modelname == 'gbm':
#         params = {}
#         # params['learning_rate'] = trial.suggest_float('learning_rate', 0.0001, 1, log=True)
#         # params['max_iter'] = trial.suggest_int('max_iter', 100, 1000, step=100)
#         # params['max_depth'] = trial.suggest_int('max_depth', 80, 110, step=10)
#         # params['l2_regularization'] = trial.suggest_float('l2_regularization', 0.01, 1, log=True)
# #        arg_features = trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2', None])
#         model = HistGradientBoostingClassifier(warm_start=True, **params)
#     elif modelname == 'xgboost':
#         # params = {}
#         # params['n_estimators'] = trial.suggest_int('n_estimators', 100, 500, step=100)
#         # params['max_depth'] = trial.suggest_int('max_depth', 3, 9, step=2)
#         # # max_leaves
#         # # max_bin
#         # # tree_method
#         # params['min_child_weight'] = trial.suggest_float("min_child_weight", 1, 6, step=1)
#         # params['learning_rate'] = trial.suggest_float('learning_rate', 0.0001, 1, log=True)
#         # # booster
#         # # n_jobs
#         # params['subsample'] = trial.suggest_float('subsample', 0.5, 0.9)
#         # params['colsample_bytree'] = trial.suggest_float('colsample_bytree', 0.5, 0.9)
#         # params['gamma'] = trial.suggest_int('gamma', 1, 9, log=True)
#         # params['reg_alpha'] = trial.suggest_float('reg_alpha', 0.00001, 1, log=True)
#         # params['reg_lambda'] = trial.suggest_float('reg_lambda', 0.00001, 1, log=True)
#         if config.args['multi-label']:
#             model = XGBClassifier(tree_method="hist", **params)
#         else:
#             model = XGBClassifier(**params)
#     elif modelname == 'mlp_keras':
#         # arg_lr = trial.suggest_float('learning_rate', 0.0001, 1, log=True)
#         epochs = 150
#         batch_size = 100
#         model = mlpclassifier_keras(train_X.shape[1:], **params)
#         model.fit(train_X, train_y, validation_split=0.2, 
#                 batch_size=batch_size, epochs=epochs)
#     elif modelname == 'catboost':
#         params = {
#             'iterations':trial.suggest_int("iterations", 1000, 20000),
#             'od_wait':trial.suggest_int('od_wait', 500, 2300),
#             'learning_rate' : trial.suggest_uniform('learning_rate',0.01, 1),
#             'reg_lambda': trial.suggest_uniform('reg_lambda',1e-5,100),
#             'subsample': trial.suggest_uniform('subsample',0,1),
#             'random_strength': trial.suggest_uniform('random_strength',10,50),
#             'depth': trial.suggest_int('depth',1, 15),
#             'min_data_in_leaf': trial.suggest_int('min_data_in_leaf',1,30),
#             'leaf_estimation_iterations': trial.suggest_int('leaf_estimation_iterations',1,15),
#             'bagging_temperature' :trial.suggest_loguniform('bagging_temperature', 0.01, 100.00),
#             'colsample_bylevel':trial.suggest_float('colsample_bylevel', 0.4, 1.0),
#         }
#         model = CatBoostClassifier(**params)
#     elif modelname == 'extratree':
#         params = {}
#         params['n_estimators'] = trial.suggest_int('n_estimators', 100, 500, step=100)
#         params['criterion'] = trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss'])
#         params['max_depth'] = trial.suggest_int('max_depth', 3, 9, step=2)
#         params['max_features'] = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
#         model = ExtraTreesClassifier(warm_start=True, **params)

#     if modelname.endswith('keras'): 
#         accuracy = model.evaluate(train_X, train_y)[-1]
        
#     else:
#         if config.args['metric'] == 'auc':
#             score = cross_val_score(
#                 model, train_X, train_y, scoring="roc_auc",
#                 n_jobs=-1, cv=3)
#         else:
#             score = cross_val_score(
#                 model, train_X, train_y, n_jobs=-1, cv=3)
#         score = score.mean()
#     return score 
