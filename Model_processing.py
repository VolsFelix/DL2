from email.header import Header
import pandas as pd
from requests import head
import numpy as np

models = pd.read_csv('models.csv', header=None)
models.head()

models=models.rename(columns=models.iloc[0]).drop(models.index[0])
models['min_val_loss'] = np.nan
models['min_train_loss'] = np.nan

models.head()
x=[]
#finding the min val loss and min train loss for each model
for j in range(1,len(models)+1):
    x.append(models['val_loss'][j][1:len(models['val_loss'][j])-1].split(','))
    for i in range(len(x)):
        for k in range(len(x[i])):
            models['min_val_loss'][j]=min(float(x[i][k]) for k in range(len(x[i])))

y=[]
for j in range(1,len(models)+1):
    y.append(models['train_loss'][j][1:len(models['train_loss'][j])-1].split(','))
    for i in range(len(y)):
        for k in range(len(x[i])):
            models['min_train_loss'][j]=min(float(y[i][k]) for k in range(len(y[i])))

models.to_csv('models_altered3.csv', index=False)