import os
from datetime import datetime as dt

import numpy as np
import pandas as pd

def submission(pred_array, columns, modelname, desc='none', dest_dir='results/'):
    df = pd.DataFrame(pred_array, columns=columns)
    now = dt.strftime(dt.now(), '%y-%m-%d')
    filename = f'{modelname}-{desc}-{now}.csv'

    i = 0
    while os.path.exists(filename):
        i += 1
        filename = f'{modelname}-{desc}-{now}-{i}.csv'
    df.to_csv(filename, index=False)

if __name__ == '__main__':
    pred_array = np.array([[1, 0.63], [10, 0.3]])
    columns = ['PassengerId', 'Survived']
    modelname = 'none'
    submission(pred_array, columns, modelname)