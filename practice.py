import pandas as pd
from itertools import product
import matplotlib.pyplot as plt

def expand_grid(dictionary):
   return pd.DataFrame([row for row in product(*dictionary.values())], 
                       columns=dictionary.keys())

dictionary = {'batch_size': [100], 
              'num_hidden_layers': [1,2,3,4,5,6,7], 
              'num_hidden_neurons': [50,100,250],
              'act_func': ["sigmoid","tanh","relu","leaky relu","prelu","elu"],
              'optimizers':["plain SGD","momentum","nesterov","adagrad","rmsprop","adam","learning rate scheduling"]}

hyper_options=expand_grid(dictionary)

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

### First step is to encode the categorical variables: category and SKU

hyper_options['batch_size'][0]
i=0
d={}

for i in range(10):
    inputs_cat = tf.keras.layers.Input(shape=(1,),name = 'in_cats')

    embedding_cat = tf.keras.layers.Embedding(input_dim=PRICING['category'].nunique()+1, output_dim=hyper_options['num_hidden_neurons'][i], input_length=1,name = 'embedding_cat')(inputs_cat)

    embedding_flat_cat = tf.keras.layers.Flatten(name='flatten')(embedding_cat)

## embedding the sku

    inputs_sku = tf.keras.layers.Input(shape=(1,),name = 'in_sku')
    embedding_sku = tf.keras.layers.Embedding(input_dim=PRICING['sku'].nunique()+1, output_dim=hyper_options['num_hidden_neurons'][i], input_length=1,name = 'embedding_sku')(inputs_sku)
    embedding_flat_sku = tf.keras.layers.Flatten(name='flatten2')(embedding_sku) 

# combining the categorical embedding layers
    cats_concat = tf.keras.layers.Concatenate(name = 'concatenation1')([embedding_flat_cat, embedding_flat_sku])

#input for the quantity, price,order, and duration
    inputs_num = tf.keras.layers.Input(shape=(3,),name = 'in_num')

#combinging the all input layers
    inputs_concat2 = tf.keras.layers.Concatenate(name = 'concatenation')([cats_concat, inputs_num])

## Hidden Layers
    hidden1 = tf.keras.layers.Dense(hyper_options['num_hidden_neurons'][i],activation=hyper_options['act_func'][i],name='hidden')(inputs_concat2)

#output layer
    outputs = tf.keras.layers.Dense(1, name = 'out')(hidden1)

    inputs=[inputs_cat,inputs_sku,inputs_num]

    model = tf.keras.Model(inputs = inputs, outputs = outputs)

    model.summary()
    model.compile(loss = 'mse', optimizer = tf.keras.optimizers.SGD(learning_rate = 0.01))




    model1=model.fit(x=input_dict,y=train['quantity'], batch_size=hyper_options['batch_size'][i], epochs=5)
    d["model{0}_loss".format(i)]=model1.history['loss']



epochs=range(0,5)

plt.plot(epochs,d['model0_loss'],label='model0')
plt.plot(epochs,d['model1_loss'],label='model1')
plt.plot(epochs,d['model2_loss'],label='model2')
plt.plot(epochs,d['model3_loss'],label='model3')
plt.plot(epochs,d['model4_loss'],label='model4')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
plt.show()


hyper_options.loc[[2]]