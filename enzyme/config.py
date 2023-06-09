import os

data_dir='/home/jylee/datasets/kaggle/playground-series-s3e18'
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
    'dataname': 'enzyme',
    'multi-label': True,
    'oversampling': True,
    'epochs': 20,
}

keras_params = {
    'learning_rate': 0.0001,
    'epochs': 500,
    'batch_size': 100,
}
