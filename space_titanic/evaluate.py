
from sklearn.metrics import accuracy_score, mean_squared_error

def evaluate(y_true, y_pred, metric='accuracy', desc='val'):
    if metric == 'accuracy':
        score = accuracy_score(y_true, y_pred)
        print(f"{desc} {metric}: ", score)
    elif metric == 'rmse':
        score = mean_squared_error(
                y_true, y_pred, squared=False)
        print(f"{desc} {metric}: ", score)
    return score
