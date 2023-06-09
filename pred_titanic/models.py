'''
Classification
- (+) logistic regression
- (+) knn (ml)
- (+) svc
- (+) adaboost 
- (+) random forest 
- (+) mlp

Regression
- ( ) linear regression
- ( ) lasso
- ( ) elasticnet
- ( ) mlp
'''
from functools import partial

import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

from sklearn.model_selection import cross_val_score
import optuna
import tensorflow as tf

from evaluate import evaluate

def get_classifier(name='knn', input_shape=None, **args):
    if name == 'knn':
        model = KNeighborsClassifier(**args)
    elif name == 'logistic':
        model = LogisticRegression(**args)
    elif name == 'svc':
        model = SVC(**args)
    elif name == 'rf':
        model = RandomForestClassifier(**args)
    elif name == 'ada':
        model = AdaBoostClassifier(**args)
    elif name == 'mlp':
        model = MLPClassifier(warm_start=True, **args)
    elif name == 'gbm':
        model = GradientBoostingClassifier(warm_start=True, **args)
    elif name == 'mlp_keras':
        learning_rate = args['learning_rate']
        model = mlpclassifier_keras(input_shape, learning_rate)

    return model


def classification_objective(trial, modelname, train_X, train_y):
#    classifier_name = trial.suggest_categorical('classifier', [])
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
        arg_loss = trial.suggest_categorical('loss', ['log_loss', 'deviance', 'exponential'])
        arg_estimators = trial.suggest_int('n_estimators', 100, 1000, step=100)
        arg_lr = trial.suggest_float('learning_rate', 0.0001, 1, log=True)
        arg_mdepth = trial.suggest_int('max_depth', 80, 110, step=10)
        arg_features = trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2', None])
        model = GradientBoostingClassifier(warm_start=True)
    elif modelname == 'mlp_keras':
        arg_lr = trial.suggest_float('learning_rate', 0.0001, 1, log=True)
#        arg_batchsize = trial.suggest_int('batch_size', 1, 1000, log=True)
#        arg_epochs = trial.suggest_int('epochs', 100, 200, log=True)
        epochs = 150
        batch_size = 20
        model = mlpclassifier_keras(train_X.shape[1:], arg_lr)
        model.fit(train_X, train_y, validation_split=0.2, 
                batch_size=batch_size, epochs=epochs)

    if modelname.endswith('keras'): 
        accuracy = model.evaluate(train_X, train_y)[-1]
        
    else:
        score = cross_val_score(
                model, train_X, train_y, n_jobs=-1, cv=5)
        accuracy = score.mean()

    return accuracy

def retrain(modelname, best_params, data_splited, test_X):
    pred_list = []
    train_accs = []
    
    if 'classifier' in best_params.keys():
        del best_params['classifier']
    
    for i in range(len(data_splited.keys())):
        if modelname.endswith('keras'):
            # learning params
            epochs = 500#best_params['epochs']
            batch_size = 20#best_params['batch_size']
#            del best_params['epochs']
#            del best_params['batch_size']

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
            model = get_classifier(modelname, **best_params)
            
            # train
            model = model.fit(data_splited[f'{i}th']['X_train'], 
                data_splited[f'{i}th']['y_train'])
        
            # evaluate
            train_pred = model.predict(data_splited[f'{i}th']['X_train'])
            val_pred = model.predict(data_splited[f'{i}th']['X_val'])
        
            train_eval = evaluate(data_splited[f'{i}th']['y_train'], train_pred, 
                     metric='accuracy', desc='train')
            _ = evaluate(data_splited[f'{i}th']['y_val'], val_pred, 
                     metric='accuracy', desc='val')
            test_pred = model.predict(test_X)
        
        # pred
        pred_list.append(test_pred)
        train_accs.append(train_eval)

    return np.array(pred_list), train_accs

def optimize(modelname, train_X, train_y):
    study = optuna.create_study(direction='maximize')
    objective = partial(
        classification_objective, modelname=modelname, 
        train_X=train_X, train_y=train_y
    )
    study.optimize(objective, n_trials=100)
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
