
from sklearn.metrics import accuracy_score

def evaluate(y_true, y_pred, metric='accuracy', desc='val'):
    accuracy = accuracy_score(y_true, y_pred)
    print(f"{desc} accuracy: ", accuracy)
    return accuracy
