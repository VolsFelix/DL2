## Group 2 Deep Learning Project
# The
# goal of this assignment is to gather experience on the sensitivity of the algorithm to different
# kinds of tuning parameters: batch size, number of hidden layers, number hidden neurons, hidden
# activation functions (sigmoid, tanh, relu, leaky relu, prelu, el
# u), optimizers (plain SGD,
# momentum, nesterov, adagrad, rmsprop, adam, learning rate scheduling

# The goal is to
# predict quantity sold of a given product as accurately as possible by tuning the learning procedure

# Kate:
# subset the data for training -- decide best size to sample : No need to do anymore, because we can train 1 epoch in just 3-4 min now
# incorporate batch reading data from csv : DONE
# test how model reacts to sku values that were not used in the training (is this similar to text mining, or is it a problem?)
#It still predict the sku values that were not used in the training

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import feature_column
from tensorflow.keras import layers
import os
import matplotlib.pyplot as plt
import time

#Change category to consecutive integer

# cond_list = [PRICING['category']<2, PRICING['category']>2, PRICING['category']==2]
# choice_list = [PRICING['category'], PRICING['category']-1, -1]
# PRICING["cat_consec"] = np.select(cond_list,choice_list)
# train, test = train_test_split(df, test_size=0.2)
# train, val = train_test_split(train, test_size=0.2)



#Load in the final file with consecutive category + sku
os.chdir('/Users/katemac2021/Documents/GitHub/DL2')
#df =  pd.read_csv("pricing_apr4.csv") #full dataset
df  = pd.read_csv("train.csv")
train =  pd.read_csv("train.csv")
#test =  pd.read_csv("test.csv")

def get_dataset(file_path):
  dataset = tf.data.experimental.make_csv_dataset(
      file_path,
      batch_size=50, 
      label_name='quantity',
      na_value="?",
      num_epochs=1,
      ignore_errors=True)
  return dataset


raw_train_data = get_dataset("train.csv") 
train_data = raw_train_data.shuffle(500)
train_data
val_data = get_dataset("val.csv")
test_data = get_dataset("test.csv")
#Test = 20% full dataset, 80% of the rest is training, 20% of the rest is validation, it is divided randomly

# numeric features being fed into the model:
feature_columns = []
feature_columns.append(tf.feature_column.numeric_column('price'))
feature_columns.append(tf.feature_column.numeric_column('order'))
feature_columns.append(tf.feature_column.numeric_column('duration'))

                       
# categorical columns using the lists created above:
sku_list = list(range(74999)) #full  dataset
cat_list = list(range(32)) #full  dataset
sku_list = train['sku'].unique()
cat_list = train['cat_consec'].unique()
sku_col = tf.feature_column.categorical_column_with_vocabulary_list(
    'sku', sku_list)
cat_col = tf.feature_column.categorical_column_with_vocabulary_list(
    'cat_consec', cat_list)
                       
# create an embedding from the categorical column:
sku_emb = tf.feature_column.embedding_column(sku_col,dimension=50) #output?, dimension?
cat_emb = tf.feature_column.embedding_column(cat_col,dimension=16) #output?, dimension?
#https://mmuratarat.github.io/2019-06-12/embeddings-with-numeric-variables-Keras#:~:text=Jeremy%20Howard%20provides%20a%20general,So%20it's%20kind%20of%20experimental.
#min of (50 and num of cat/2) is the best number of layer
# add the embeddings to the list of feature columns
feature_columns.append(sku_emb)
feature_columns.append(cat_emb)
                       
# create the input layer for the model

feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

model = tf.keras.models.Sequential()
model.add(feature_layer)
#model.add(tf.keras.layers.Dropout(0.25)) #Now we dont do dropout

he_avg_init = tf.keras.initializers.VarianceScaling(scale = 2., mode ="fan_avg", distribution = 'uniform')

for units in [30,15,6]:
  dense_tensor = tf.keras.layers.Dense(units, activation='ELU',kernel_initializer =he_avg_init)
  model.add(dense_tensor)

model.add(layers.Dense(1))

model.compile(loss = 'mse', optimizer = tf.keras.optimizers.Adam(learning_rate=0.01))
start = time.time()
model.fit(train_data, epochs=1)
time.time()-start
model.summary()

# model.evaluate(val_data)
# predict = model.evaluate(test_data)

#   # #Evaluate: Train
# sum(np.isnan(predict))

# results_train = model.evaluate()
# print("test loss, test acc:", results_train)



# results_test = model.evaluate(x=[x_cat_list_t,x_num_t],y=y_t, batch_size=128)
# print("test loss, test acc:", results_test)



# #history3 = model.fit(x=X,y=y, batch_size=50, epochs=10)

# pd.DataFrame(history3.history).plot(figsize=(8,5))
# plt.show()