import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

All_models= pd.read_csv('models_altered.csv')

# bar chart of hidden layers vs min_val_loss by activation function
plt.figure(figsize=(10,5))
elu=All_models[All_models['activation_function']=='elu']
best_elu=elu.sort_values(by='min_val_loss').iloc[0]

sig=All_models[All_models['activation_function']=='sigmoid']
best_sig=sig.sort_values(by='min_val_loss').iloc[0]

tanh=All_models[All_models['activation_function']=='tanh']
best_tanh=tanh.sort_values(by='min_val_loss').iloc[0]

plt.bar(best_elu['hidden_layers'], best_elu['min_val_loss'],color='r',label='elu')
plt.bar(best_sig['hidden_layers'], best_sig['min_val_loss'],color='b',label='sigmoid')
plt.bar(best_tanh['hidden_layers'], best_tanh['min_val_loss'],color='g',label='tanh')
plt.legend(['elu','sigmoid','tanh'])
plt.xlabel('Number of Hidden Layers')
plt.ylabel('Min Val Loss')
plt.title('Min Val Loss by Activation Function')
plt.show()


