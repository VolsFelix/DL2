## Group 2 Deep Learning Project
# The
# goal of this assignment is to gather experience on the sensitivity of the algorithm to different
# kinds of tuning parameters: batch size, number of hidden layers, number hidden neurons, hidden
# activation functions (sigmoid, tanh, relu, leaky relu, prelu, el
# u), optimizers (plain SGD,
# momentum, nesterov, adagrad, rmsprop, adam, learning rate scheduling

# The goal is to
# predict quantity sold of a given product as accurately as possible by tuning the learning procedure

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import feature_column
from tensorflow.keras import layers



PRICING= pd.read_csv('pricing.csv')
PRICING.head()

## splitting the data into test and train sets
train, test = train_test_split(PRICING, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)


### First step is to encode the categorical variables: category and SKU



#######################
#one categorical feature (one-hot) and one numeric feature
#Embeddings

#consider the following data where integers represent categories:
x_cat = PRICING['category']
x_num = PRICING['price'] #numeric
#random y data
y = PRICING['quantity']


inputs_cat = tf.keras.layers.Input(shape=(1,),name = 'in_cat')
embedding = tf.keras.layers.Embedding(input_dim=len(x_cat.unique()), output_dim=3, input_length=1,name = 'embedding')(inputs_cat)
#Embedding shape is (None, 1, 3)
#Need to flatten to make shape compatible with numeric input, which only has two dimensions: (None, 1)
embedding_flat = tf.keras.layers.Flatten(name='flatten')(embedding) 

#input for the integer numbers
inputs_num = tf.keras.layers.Input(shape=(1,),name = 'in_num')

#input for embedding sku. Commented out for now Cause i Couldnt get it to fit in with everything else
# inputs_sku = tf.keras.layers.Input(shape=(1,),name = 'in_sku')
# embedding_sku = tf.keras.layers.Embedding(input_dim=len(PRICING['sku'].unique()), output_dim=3, input_length=1,name = 'embedding_sku')(inputs_sku)
# embedding_flat_sku = tf.keras.layers.Flatten(name='flatten_sku')(embedding_sku)

#combinging the input layers
inputs_concat = tf.keras.layers.Concatenate(name = 'concatenation')([embedding_flat, inputs_num])
hidden = tf.keras.layers.Dense(2,name='hidden')(inputs_concat)
outputs = tf.keras.layers.Dense(1, name = 'out')(hidden)


model = tf.keras.Model(inputs = [inputs_cat,inputs_num], outputs = outputs)
model.summary()
model.compile(loss = 'mse', optimizer = tf.keras.optimizers.SGD(learning_rate = 0.001))
model.fit(x=[x_cat,x_num],y=y, batch_size=100, epochs=1)





