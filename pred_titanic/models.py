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

from sklearn.neighbors import KNeighborsClassifier


## params
params = {
    'knn': {
        'n_neghbors': 5,
    }
}

def get_classifier(name='knn'):
    if name == 'knn':
        model = KNeighborsClassifier(params['knn'])
    return model

## optuma 