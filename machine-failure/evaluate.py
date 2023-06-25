from sklearn.metrics import (
    accuracy_score, mean_squared_error, roc_auc_score,
    auc
)

def evaluate(y_true, y_pred, desc='val'):
    if metric == 'acc':
        score = accuracy_score(y_true, y_pred)
        print(f"{desc} {metric}: ", score)
    elif metric == 'auc':
        fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=2)
        score = auc(fpr, tpr)
        print(f"{desc} {metric}: ", score)
    elif metric == 'rmse':
        score = mean_squared_error(
                y_true, y_pred, squared=False)
        print(f"{desc} {metric}: ", score)
    return score
