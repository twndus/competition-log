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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import optuna

## params
params = {
    'knn': {
        'n_neghbors': 5,
    }
}

def get_classifier(name='knn'):
    if name == 'knn':
        model = KNeighborsClassifier()
    return model


def objective1(trial, train_X, train_y):
    classifier_name = trial.suggest_categorical('classifier', ['knn'])
    if classifier_name == 'knn':
        knn_n = trial.suggest_int('knn_n', 3, 9)
        classifier_obj = KNeighborsClassifier()
    
    score = cross_val_score(
            classifier_obj, train_X, train_y, n_jobs=-1, cv=3)
    accuracy = score.mean()
    return accuracy

def optimize(train_X, train_y):
    study = optuna.create_study(direction='maximize')
    objective = partial(objective1, train_X=train_X, train_y=train_y)
    study.optimize(objective, n_trials=100)
    print(study.best_trial)

