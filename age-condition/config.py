import os

data_dir='/home/jylee/datasets/kaggle/icr-identify-age-related-conditions'

# args
args = {
    'train_path': os.path.join(data_dir, 'train.csv'),
    'test_path': os.path.join(data_dir, 'test.csv'),
    'submission_path': os.path.join(data_dir, 'sample_submission.csv'),
    'task': 'classification',
    'modelname': 'catboost',
    'metric': 'balanced_log_loss',
    #'xgboost' #'mlp_keras' #'gbm'#'mlp' #'rf' #'svc' #'logistic' #'knn'
    'dataname': 'age',
    'epochs': 20,
}

