classification_params = {
    'knn': {
        'n_neighbors': {'type': 'cat', 'continuous': False, 'values': [3, 5, 7]},
    },
    'logistic': {
        'penalty': {'type': 'cat', 'continuous': False, 'values': ['l1', 'l2', 'elasticnet']},
        'tol': {'type': 'float','continuous': True, 'values': [1e-5, 1e-2, None, True]},
    },
    'svc': {
        'C': {'type': 'float','continuous': True, 'values': [0.1, 1000, None, True]},
        'gamma': {'type': 'float','continuous': True, 'values': [0.0001, 1, None, True]},
    },
    'rf': {
        'criterion': {'type': 'cat', 'continuous': False, 'values': ['gini', 'entropy']},
        'max_depth': {'type': 'int', 'continuous': True, 'values': [80, 110, 10, False]},
        'max_features': {'type': 'int', 'continuous': True, 'values': [2, 4, 1, False]},
        'n_estimators': {'type': 'int', 'continuous': True, 'values': [50, 500, 100, False]},
    },
    'ada': {
        'n_estimators': {'type': 'int', 'continuous': True, 'values': [100, 1000, 100, False]},
        'learning_rate': {'type': 'float','continuous': True, 'values': [0.0001, 1, None, True]},
        'algorithm': {'type': 'cat', 'continuous': False, 'values': ['SAMME', 'SAMME.R']},
    },
    'mlp': {
        'learning_rate_init': {'type': 'float','continuous': True, 'values': [0.0001, 1, None, True]},
        'learning_rate': {'type': 'cat', 'continuous': False, 'values': ['constant', 'invscaling', 'adaptive']},
        'alpha': {'type': 'float','continuous': True, 'values': [0.0001, 1, 1, True]},
        'activation': {'type': 'cat', 'continuous': False, 'values': ['logistic', 'relu', 'tanh']},
        'learning_rate': {'type': 'cat', 'continuous': False, 'values': ['constant', 'invscaling', 'adaptive']},
        'batch_size': {'type': 'int', 'continuous': True, 'values': [1, 1000, 100, True]},
        'hidden_layer_sizes': {'type': 'int', 'continuous': True, 'values': [1, 1000, 1, True]},
        'max_iter': {'type': 'int', 'continuous': True, 'values': [100, 1000, 100, False]},
    },
    'gbm': {
        'learning_rate': {'type': 'float','continuous': True, 'values': [0.0001, 1, 1, True]},
        'max_iter': {'type': 'int', 'continuous': True, 'values': [100, 1000, 100, False]},
        'max_depth': {'type': 'int', 'continuous': True, 'values': [80, 110, 10, False]},
        'l2_regularization': {'type': 'float','continuous': True, 'values': [0.001, 10, 1, True]},
    },
    'xgboost': {
        'n_estimators': {'type': 'int', 'continuous': True, 'values': [50, 500, 100, False]},
        'max_depth': {'type': 'int', 'continuous': True, 'values': [80, 110, 10, False]},
        'min_child_weight': {'type': 'float','continuous': True, 'values': [1, 6, 1, False]},
        'learning_rate': {'type': 'float','continuous': True, 'values': [0.0001, 1, None, True]},
        'subsample': {'type': 'float','continuous': True, 'values': [0.5, 0.9, None, False]},
        'colsample_bytree': {'type': 'float','continuous': True, 'values': [0.5, 0.9, None, False]},
        'gamma': {'type': 'int', 'continuous': True, 'values': [1, 9, None, True]},
        'reg_alpha': {'type': 'float','continuous': True, 'values': [0.00001, 1, None, True]},
        'reg_lambda': {'type': 'float','continuous': True, 'values': [0.00001, 1, None, True]},
    },
    'mlp_keras': {
        'learning_rate': {'type': 'float','continuous': True, 'values': [0.0001, 1, None, True]},
    },
    'catboost': {
        'iterations': {'type': 'int', 'continuous': True, 'values': [1000, 20000, None, False]},
        'od_wait': {'type': 'int', 'continuous': True, 'values': [500, 2300, None, False]},
        'learning_rate': {'type': 'float','continuous': True, 'values': [0.0001, 1, None, True]},
        'reg_lambda': {'type': 'float','continuous': True, 'values': [1e-5,100, None, True]},
        'subsample': {'type': 'float','continuous': True, 'values': [0, 1, 0.1]},
        'random_strength': {'type': 'float','continuous': True, 'values': [10, 50, 10, False]},
        'depth': {'type': 'int', 'continuous': True, 'values': [1, 15, 1, False]},
        'min_data_in_leaf': {'type': 'int', 'continuous': True, 'values': [1, 30, None, False]},
        'leaf_estimation_iterations': {'type': 'int', 'continuous': True, 'values': [1, 15, None, False]},
        'bagging_temperature': {'type': 'float','continuous': True, 'values': [0.01, 100, None, True]},
        'colsample_bylevel': {'type': 'float','continuous': True, 'values': [0.4, 1.0, 0.1, False]},
    },
    'extratree': {
        'n_estimators': {'type': 'int', 'continuous': True, 'values': [50, 500, 100, False]},
        'criterion': {'type': 'cat', 'continuous': False, 'values': ['gini', 'entropy', 'log_loss']},
        'max_depth': {'type': 'int', 'continuous': True, 'values': [3, 9, 2, False]},
        'max_features': {'type': 'cat', 'continuous': False, 'values': ['sqrt', 'log2', None]},
        'od_wait': {'type': 'int', 'continuous': True, 'values': [500, 2300, None, False]},
        'learning_rate': {'type': 'float','continuous': True, 'values': [0.0001, 1, None, True]},
        'reg_lambda': {'type': 'float','continuous': True, 'values': [1e-5,100, None, True]},
        'subsample': {'type': 'float','continuous': True, 'values': [0, 1, 0.1]},
        'random_strength': {'type': 'float','continuous': True, 'values': [10, 50, 10, False]},
        'depth': {'type': 'int', 'continuous': True, 'values': [1, 15, 1, False]},
        'min_data_in_leaf': {'type': 'int', 'continuous': True, 'values': [1, 30, None, False]},
        'leaf_estimation_iterations': {'type': 'int', 'continuous': True, 'values': [1, 15, None, False]},
        'bagging_temperature': {'type': 'float','continuous': True, 'values': [0.01, 100, None, True]},
        'colsample_bylevel': {'type': 'float','continuous': True, 'values': [0.4, 1.0, 0.1, False]},
    },
}

regression_params = {
    'knn': {
        'n_neighbors': {'type': 'cat', 'continuous': False, 'values': [3, 5, 7]},
    },
    'rf': {
        'max_depth': {'type': 'int', 'continuous': True, 'values': [80, 110, 10, False]},
        'max_features': {'type': 'int', 'continuous': True, 'values': [2, 4, 1, False]},
        'n_estimators': {'type': 'int', 'continuous': True, 'values': [100, 1000, 100, False]},
    },
    'mlp': {
        'learning_rate_init': {'type': 'float','continuous': True, 'values': [0.0001, 1, None, True]},
        'learning_rate': {'type': 'cat', 'continuous': False, 'values': ['constant', 'invscaling', 'adaptive']},
        'alpha': {'type': 'float','continuous': True, 'values': [0.0001, 1, 1, True]},
        'activation': {'type': 'cat', 'continuous': False, 'values': ['logistic', 'relu', 'tanh']},
        'solver': {'type': 'cat', 'continuous': False, 'values': ['lbfgs', 'sgd', 'adam']},
        'batch_size': {'type': 'int', 'continuous': True, 'values': [1, 1000, 100, True]},
        'hidden_layer_sizes': {'type': 'int', 'continuous': True, 'values': [1, 1000, 1, True]},
        'max_iter': {'type': 'int', 'continuous': True, 'values': [100, 1000, 100, False]},
    },
    'gbm': {
        'loss': {'type': 'cat', 'continuous': False, 'values': ['squared_error', 'absolute_error', 
            'huber', 'quantile']},
        'learning_rate': {'type': 'float','continuous': True, 'values': [0.0001, 1, None, True]},
        'max_depth': {'type': 'int', 'continuous': True, 'values': [80, 110, 10, False]},
    },
    'ada': {
        'n_estimators': {'type': 'int', 'continuous': True, 'values': [100, 1000, 100, False]},
        'learning_rate': {'type': 'float','continuous': True, 'values': [0.0001, 1, 1, True]},
        'loss': {'type': 'cat', 'continuous': False, 'values': ['log_loss', 'deviance', 'exponential']},
    }
}

# classification_params = {
#     'knn': {
#         'n_neighbors': {'type': 'cat', 'continuous': False, 'values': [3, 5, 7]},
#     },
#     'logistic': {
#         'penalty': {'type': 'cat', 'continuous': False, 'values': ['l1', 'l2', 'elasticnet']},
#         'tol': {'type': 'float','continuous': True, 'values': [1e-5, 1e-2, None, True]},
#     },
#     'svc': {
#         'C': {'type': 'float','continuous': True, 'values': [0.1, 1000, None, True]},
#         'gamma': {'type': 'float','continuous': True, 'values': [0.0001, 1, None, True]},
#     },
#     'rf': {
#         'criterion': {'type': 'cat', 'continuous': False, 'values': ['gini', 'entropy']},
#         'max_depth': {'type': 'int', 'continuous': True, 'values': [80, 110, 10, False]},
#         'max_features': {'type': 'int', 'continuous': True, 'values': [2, 4, 1, False]},
#         'n_estimators': {'type': 'int', 'continuous': True, 'values': [50, 500, 100, False]},
#     },
#     'ada': {
#         'n_estimators': {'type': 'int', 'continuous': True, 'values': [100, 1000, 100, False]},
#         'learning_rate': {'type': 'float','continuous': True, 'values': [0.0001, 1, None, True]},
#         'algorithm': {'type': 'cat', 'continuous': False, 'values': ['SAMME', 'SAMME.R']},
#     },
#     'mlp': {
#         'learning_rate_init': {'type': 'float','continuous': True, 'values': [0.0001, 1, None, True]},
#         'learning_rate': {'type': 'cat', 'continuous': False, 'values': ['constant', 'invscaling', 'adaptive']},
#         'alpha': {'type': 'float','continuous': True, 'values': [0.0001, 1, 1, True]},
#         'activation': {'type': 'cat', 'continuous': False, 'values': ['logistic', 'relu', 'tanh']},
#         'learning_rate': {'type': 'cat', 'continuous': False, 'values': ['constant', 'invscaling', 'adaptive']},
#         'batch_size': {'type': 'int', 'continuous': True, 'values': [1, 1000, 100, True]},
#         'hidden_layer_sizes': {'type': 'int', 'continuous': True, 'values': [1, 1000, 1, True]},
#         'max_iter': {'type': 'int', 'continuous': True, 'values': [100, 1000, 100, False]},
#     },
#     'gbm': {
#         'learning_rate': {'type': 'float','continuous': True, 'values': [0.0001, 1, 1, True]},
#         'max_iter': {'type': 'int', 'continuous': True, 'values': [100, 1000, 100, False]},
#         'max_depth': {'type': 'int', 'continuous': True, 'values': [80, 110, 10, False]},
#         'l2_regularization': {'type': 'float','continuous': True, 'values': [0.001, 10, 1, True]},
#     },
#     'xgboost': {
#         'n_estimators': {'type': 'int', 'continuous': True, 'values': [50, 500, 100, False]},
#         'max_depth': {'type': 'int', 'continuous': True, 'values': [80, 110, 10, False]},
#         'min_child_weight': {'type': 'float','continuous': True, 'values': [1, 6, 1, False]},
#         'learning_rate': {'type': 'float','continuous': True, 'values': [0.0001, 1, None, True]},
#         'subsample': {'type': 'float','continuous': True, 'values': [0.5, 0.9, None, False]},
#         'colsample_bytree': {'type': 'float','continuous': True, 'values': [0.5, 0.9, None, False]},
#         'gamma': {'type': 'int', 'continuous': True, 'values': [1, 9, None, True]},
#         'reg_alpha': {'type': 'float','continuous': True, 'values': [0.00001, 1, None, True]},
#         'reg_lambda': {'type': 'float','continuous': True, 'values': [0.00001, 1, None, True]},
#     },
#     'mlp_keras': {
#         'learning_rate': {'type': 'float','continuous': True, 'values': [0.0001, 1, None, True]},
#     }
# }
