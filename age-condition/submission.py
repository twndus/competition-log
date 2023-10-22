import os
from datetime import datetime as dt

import numpy as np
import pandas as pd

def submission(pred_array, submission_df, modelname, 
        train_metrics='none', desc='none', dest_dir='results/', sub_type='float'):

    submission_df.iloc[:, 1] = pred_array
    
    if sub_type == 'bool':
        submission_df.iloc[:, 1] = \
            submission_df.iloc[:, 1].astype('boolean')

    now = dt.strftime(dt.now(), '%y-%m-%d')
    filename = f'{modelname}-train-{train_metrics}-{desc}-{now}.csv'

    i = 0
    while os.path.exists(os.path.join(dest_dir, filename)):
        i += 1
        filename = f'{modelname}-train-{train_metrics}-{desc}-{now}-{i}.csv'
    submission_df.to_csv(os.path.join(dest_dir, filename), index=False)

if __name__ == '__main__':
    pred_array = np.random.normal(0, 1, 418)
    submission_path = '../../../datasets/kaggle/titanic/gender_submission.csv'
    submission_df = pd.read_csv(submission_path)
    modelname = 'none'
    submission(pred_array, submission_df, modelname)
