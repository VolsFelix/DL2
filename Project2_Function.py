## Group 2 Deep Learning Project
# The goal of this assignment is to gather experience on the sensitivity of the algorithm to different
# kinds of tuning parameters: batch size, number of hidden layers, number hidden neurons, hidden
# activation functions (sigmoid, tanh, relu, leaky relu, prelu, elu),
# optimizers (plain SGD, momentum, nesterov, adagrad, rmsprop, adam, learning rate scheduling)

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
inputs_cat = tf.keras.layers.Input(shape=(1,),name = 'in_cats')
# embedding category
embedding_cat = tf.keras.layers.Embedding(input_dim=train['category'].nunique()+1, output_dim=3, input_length=1,name = 'embedding_cat')(inputs_cat)
embedding_flat_cat = tf.keras.layers.Flatten(name='flatten')(embedding_cat)

# embedding the sku
inputs_sku = tf.keras.layers.Input(shape=(1,),name = 'in_sku')
embedding_sku = tf.keras.layers.Embedding(input_dim=PRICING['sku'].nunique(), output_dim=3, input_length=1,name = 'embedding_sku')(inputs_sku)
embedding_flat_sku = tf.keras.layers.Flatten(name='flatten2')(embedding_sku)

# combining the categorical embedding layers
cats_concat = tf.keras.layers.Concatenate(name = 'concatenation1')([embedding_flat_cat, embedding_flat_sku])

#input for the quantity, price,order, and duration
inputs_num = tf.keras.layers.Input(shape=(3,),name = 'in_num')

#combinging the all input layers
inputs_concat2 = tf.keras.layers.Concatenate(name = 'concatenation')([cats_concat, inputs_num])

## Hidden Layers
hidden1 = tf.keras.layers.Dense(2,activation='sigmoid',name='hidden')(inputs_concat2)

#output layer
outputs = tf.keras.layers.Dense(1, name = 'out')(hidden1)

inputs=[inputs_cat,inputs_sku,inputs_num]

model = tf.keras.Model(inputs = inputs, outputs = outputs)

model.summary()
model.compile(loss = 'mse', optimizer = tf.keras.optimizers.SGD(learning_rate = 0.01))


## seperating the numerical features from rest of dataset
num_features=train.drop(['sku'], axis=1)
num_features=num_features.drop(['category'], axis=1)
num_features=num_features.drop(['quantity'], axis=1)

## creates an input dictionary for the model
input_dict= {
    'in_cats':train["category"],
    "in_sku":train["sku"],
    "in_num": num_features
}

model.fit(x=input_dict,y=train['quantity'], batch_size=50, epochs=1)





