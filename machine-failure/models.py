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
    )
from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from sklearn.model_selection import cross_val_score
import optuna
import tensorflow as tf

from evaluate import evaluate
from config import args

metrics = {'classification': 'accuracy',
        'regression': 'rmse'}

def get_classifier(modelname='knn', task=None, input_shape=None, **args):
    if task == 'classification':
        if modelname == 'knn':
            model = KNeighborsClassifier(**args)
        elif modelname == 'logistic':
            model = LogisticRegression(**args)
        elif modelname == 'svc':
            model = SVC(**args)
        elif modelname == 'rf':
            model = RandomForestClassifier(**args)
        elif modelname == 'ada':
            model = AdaBoostClassifier(**args)
        elif modelname == 'mlp':
            model = MLPClassifier(warm_start=True, **args)
        elif modelname == 'gbm':
            model = HistGradientBoostingClassifier(warm_start=True, **args)
        elif modelname == 'mlp_keras':
            learning_rate = args['learning_rate']
            model = mlpclassifier_keras(input_shape, learning_rate)
        elif modelname == 'xgboost':
            model = XGBClassifier(**args)
        elif modelname == 'catboost':
            model = CatBoostClassifier(**args)
    elif task == 'regression':
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
#            model = AdaBoostRegressor(**args)
    return model


def classification_objective(trial, modelname, train_X, train_y):
    classifier_name = trial.suggest_categorical('classifier', [modelname])
    if modelname == 'knn':
        arg_n = trial.suggest_categorical('n_neighbors', [3, 5, 7])
        model = KNeighborsClassifier()
    elif modelname == 'logistic':
        arg_p = trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet'])
        arg_tol = trial.suggest_float('tol', 1e-5, 1e-2)
        model = LogisticRegression()
    elif modelname == 'svc':
        arg_c = trial.suggest_float('C', 0.1, 1000, log=True)
        arg_gamma = trial.suggest_float('gamma', 0.0001, 1, log=True)
        model = SVC()
    elif modelname == 'rf':
        arg_mdepth = trial.suggest_int('max_depth', 80, 110, step=10)
        arg_features = trial.suggest_int('max_features', 2, 4)
        arg_estimators = trial.suggest_int('n_estimators', 100, 1000, step=100)
        model = RandomForestClassifier()
    elif modelname == 'ada':
        arg_estimators = trial.suggest_int('n_estimators', 100, 1000, step=100)
        arg_lr = trial.suggest_float('learning_rate', 0.0001, 1, log=True)
        arg_algorithms = trial.suggest_categorical('algorithm', ['SAMME', 'SAMME.R'])
        model = AdaBoostClassifier()
    elif modelname == 'mlp':
        arg_lrinit = trial.suggest_float('learning_rate_init', 0.0001, 1, log=True)
        arg_lr = trial.suggest_categorical('learning_rate', ['constant', 'invscaling', 'adaptive'])
        arg_alpha = trial.suggest_float('alpha', 0.0001, 1, log=True)
        arg_activation = trial.suggest_categorical('activation', ['logistic', 'relu', 'tanh'])
        arg_solver = trial.suggest_categorical('solver', ['lbfgs', 'sgd', 'adam'])
        arg_batchsize = trial.suggest_int('batch_size', 1, 1000, log=True)
        arg_hidden_layer_sizes = trial.suggest_int('hidden_layer_sizes', 1, 1000)
        arg_maxiter = trial.suggest_int('max_iter', 100, 1000)
        model = MLPClassifier(warm_start=True)
    elif modelname == 'gbm':
        params = {}
        params['learning_rate'] = trial.suggest_float('learning_rate', 0.0001, 1, log=True)
        params['max_iter'] = trial.suggest_int('max_iter', 100, 1000, step=100)
        params['max_depth'] = trial.suggest_int('max_depth', 80, 110, step=10)
        params['l2_regularization'] = trial.suggest_float('l2_regularization', 0.01, 1, log=True)
#        arg_features = trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2', None])
        model = HistGradientBoostingClassifier(warm_start=True, **params)
    elif modelname == 'xgboost':
        params = {}
        params['n_estimators'] = trial.suggest_int('n_estimators', 100, 500, step=100)
        params['max_depth'] = trial.suggest_int('max_depth', 3, 9, step=2)
        # max_leaves
        # max_bin
        # tree_method
        params['min_child_weight'] = trial.suggest_float("min_child_weight", 1, 6, step=1)
        params['learning_rate'] = trial.suggest_float('learning_rate', 0.0001, 1, log=True)
        # booster
        # n_jobs
        params['subsample'] = trial.suggest_float('subsample', 0.5, 0.9)
        params['colsample_bytree'] = trial.suggest_float('colsample_bytree', 0.5, 0.9)
        params['gamma'] = trial.suggest_int('gamma', 1, 9, log=True)
        params['reg_alpha'] = trial.suggest_float('reg_alpha', 0.00001, 1, log=True)
        params['reg_lambda'] = trial.suggest_float('reg_lambda', 0.00001, 1, log=True)
        model = XGBClassifier(**params)
    elif modelname == 'mlp_keras':
        arg_lr = trial.suggest_float('learning_rate', 0.0001, 1, log=True)
        epochs = 150
        batch_size = 20
        model = mlpclassifier_keras(train_X.shape[1:], arg_lr)
        model.fit(train_X, train_y, validation_split=0.2, 
                batch_size=batch_size, epochs=epochs)
    elif modelname == 'catboost':
        # params = {}
        # params['iterations']=trial.suggest_int("iterations", 100, 1000)
        # params['learning_rate']=trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
        # params['depth']=trial.suggest_int("depth", 4, 10)
        # params['l2_leaf_reg']=trial.suggest_float("l2_leaf_reg", 1e-8, 100.0, log=True)
        # params['bootstrap_type']=trial.suggest_categorical("bootstrap_type", ["Bayesian"])
        # params['random_strength']=trial.suggest_float("random_strength", 1e-8, 10.0, log=True)
        # params['bagging_temperature']=trial.suggest_float("bagging_temperature", 0.0, 10.0)
        # params['od_type']=trial.suggest_categorical("od_type", ["IncToDec", "Iter"])
        # params['od_wait']=trial.suggest_int("od_wait", 10, 50)
        # params['verbose']=False 
        # params['min_child_samples'] = trial.suggest_categorical('min_child_samples', [1, 4, 8, 16, 32])
        params = {
            'iterations':trial.suggest_int("iterations", 1000, 20000),
            'od_wait':trial.suggest_int('od_wait', 500, 2300),
            'learning_rate' : trial.suggest_uniform('learning_rate',0.01, 1),
            'reg_lambda': trial.suggest_uniform('reg_lambda',1e-5,100),
            'subsample': trial.suggest_uniform('subsample',0,1),
            'random_strength': trial.suggest_uniform('random_strength',10,50),
            'depth': trial.suggest_int('depth',1, 15),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf',1,30),
            'leaf_estimation_iterations': trial.suggest_int('leaf_estimation_iterations',1,15),
            'bagging_temperature' :trial.suggest_loguniform('bagging_temperature', 0.01, 100.00),
            'colsample_bylevel':trial.suggest_float('colsample_bylevel', 0.4, 1.0),
        }
        model = CatBoostClassifier(**params)

    if modelname.endswith('keras'): 
        accuracy = model.evaluate(train_X, train_y)[-1]
        
    else:
        if args['metric'] == 'auc':
            score = cross_val_score(
                model, train_X, train_y, scoring="roc_auc",
                n_jobs=-1, cv=3)
        else:
            score = cross_val_score(
                model, train_X, train_y, n_jobs=-1, cv=3)
        score = score.mean()
    return score 

def regression_objective(trial, modelname, train_X, train_y):
    regression_name= trial.suggest_categorical('classifier', [modelname])
    if modelname == 'knn':
        arg_n = trial.suggest_categorical('n_neighbors', [3, 5, 7])
        model = KNeighborsRegressor()
    elif modelname == 'rf':
        arg_mdepth = trial.suggest_int('max_depth', 80, 110, step=10)
        arg_features = trial.suggest_int('max_features', 2, 4)
        arg_estimators = trial.suggest_int('n_estimators', 100, 1000, step=100)
        model = RandomForestRegressor()
    elif modelname == 'mlp':
        arg_lrinit = trial.suggest_float('learning_rate_init', 0.0001, 1, log=True)
        arg_lr = trial.suggest_categorical('learning_rate', ['constant', 'invscaling', 'adaptive'])
        arg_alpha = trial.suggest_float('alpha', 0.0001, 1, log=True)
        arg_activation = trial.suggest_categorical('activation', ['logistic', 'relu', 'tanh'])
        arg_solver = trial.suggest_categorical('solver', ['lbfgs', 'sgd', 'adam'])
        arg_batchsize = trial.suggest_int('batch_size', 1, 1000, log=True)
        arg_hidden_layer_sizes = trial.suggest_int('hidden_layer_sizes', 1, 1000)
        arg_maxiter = trial.suggest_int('max_iter', 100, 1000)
        model = MLPRegressor(warm_start=True)
    elif modelname == 'gbm':
        arg_loss = trial.suggest_categorical('loss', ['squared_error', 'absolute_error', 
            'huber', 'quantile'])
        arg_lr = trial.suggest_float('learning_rate', 0.0001, 1, log=True)
        arg_mdepth = trial.suggest_int('max_depth', 80, 110, step=10)
        model = HistGradientBoostingRegressor(warm_start=True)
    elif modelname == 'ada':
        arg_estimators = trial.suggest_int('n_estimators', 100, 1000, step=100)
        arg_lr = trial.suggest_float('learning_rate', 0.0001, 1, log=True)
        arg_loss = trial.suggest_categorical('loss', ['log_loss', 'deviance', 'exponential'])
        model = AdaBoostRegressor(DecisionTreeRegressor())
    
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
            model = get_classifier(modelname, input_shape=test_X.shape[1:], **best_params)
            
            history = model.fit(
                data_splited[f'{i}th']['X_train'], 
                data_splited[f'{i}th']['y_train'], epochs=epochs, 
                validation_data=(data_splited[f'{i}th']['X_val'], 
                data_splited[f'{i}th']['y_val']),
                    batch_size=batch_size)
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

def mlpclassifier_keras(input_shape, learning_rate):

    # build model
    input_layer = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Dense(units=100, activation='relu')(input_layer)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(units=1, activation='sigmoid')(x)
    model = tf.keras.Model(input_layer, x)

    # optimizer
    adam = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # compile model
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['acc'])

    return model
