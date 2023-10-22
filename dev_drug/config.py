import os

data_dir='data'
# args
args = {
    'train_path': os.path.join(data_dir, 'train.csv'),
    'test_path': os.path.join(data_dir, 'test.csv'),
    'submission_path': os.path.join(data_dir, 'sample_submission.csv'),
    'task': 'classification',
#    'modelname': 'lgbm+mlp',
    'modelname': 'mlp_keras',
    'metric': 'auc',
    #'xgboost' #'mlp_keras' #'gbm'#'mlp' #'rf' #'svc' #'logistic' #'knn'
    'dataname': 'drug',
    'multi-label': True,
    'oversampling': True,
    'epochs': 20,
}

keras_params = {
    'learning_rate': 0.0001,
    'epochs': 500,
    'batch_size': 100,
}
