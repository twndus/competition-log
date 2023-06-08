
from sklearn.metrics import accuracy_score

def evaluate(y_true, y_pred, metric='accuracy', desc='val'):
    # train_pred = model.predict(
    #     data_splited['1th']['X_train'],
    # )

    accuracy = accuracy_score(y_true, y_pred)


    # val_pred = model.predict(
    #     data_splited['1th']['X_val'],
    # )

    # val_accuracy = accuracy_score(
    #     data_splited['1th']['y_val'],
    #     val_pred)
    
    print(f"{desc} accuracy: ", accuracy)
    # print("val accuracy: ", val_accuracy)