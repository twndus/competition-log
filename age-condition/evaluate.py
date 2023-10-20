from sklearn.metrics import (
    accuracy_score, mean_squared_error, roc_auc_score,
    log_loss,
)
from config import args 

def evaluate(y_true, y_pred, desc='val'):
    # classification
    if args['metric'] == 'acc':
        score = accuracy_score(y_true, y_pred)
        print(f"{desc} {args['metric']}: ", score)
    elif args['metric'] == 'auc':
        score = roc_auc_score(y_true, y_pred)
        print(f"{desc} {args['metric']}: ", score)
    elif args['metric'] == 'balanced_log_loss':
        score = log_loss(y_true, y_pred) / 2
        print(f"{desc} {args['metric']}: ", score)
    # regression
    elif args['metric'] == 'rmse':
        score = mean_squared_error(
                y_true, y_pred, squared=False)
        print(f"{desc} {args['metric']}: ", score)
    return score
