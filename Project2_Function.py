#### Group 2 Deep Learning Project
# The goal of this assignment is to gather experience on the sensitivity of the algorithm to different
# kinds of tuning parameters: batch size, number of hidden layers, number hidden neurons, hidden
# activation functions (sigmoid, tanh, relu, leaky relu, prelu, elu),
# optimizers (plain SGD, momentum, nesterov, adagrad, rmsprop, adam, learning rate scheduling)

# The goal is to predict quantity sold of a given product
# as accurately as possible by tuning the learning procedure

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import feature_column
from tensorflow.keras import layers


PRICING= pd.read_csv('pricing.csv')
PRICING.head()

# Check if the values are consecutively encoded:
def checkConsecutive(l):
    return sorted(l) == list(range(min(l), max(l) + 1))
print(checkConsecutive(np.unique(PRICING['sku'])))
print(checkConsecutive(np.unique(PRICING['category'])))

# Change category to consecutive integer
cond_list = [PRICING['category']<2, PRICING['category']>2, PRICING['category']==2]
choice_list = [PRICING['category'], PRICING['category']-1, -1]
PRICING["category"] = np.select(cond_list,choice_list)
# Confirm
print(checkConsecutive(np.unique(PRICING['category'])))

## splitting the data into test and train sets
train, test = train_test_split(PRICING, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)


## First step is to encode the categorical variables: category and SKU
# embedding category
inputs_cat = tf.keras.layers.Input(shape=(1,),name = 'in_cats')
embedding_cat = tf.keras.layers.Embedding(input_dim=PRICING['category'].nunique()+1, output_dim=16, input_length=1,name = 'embedding_cat')(inputs_cat)
embedding_flat_cat = tf.keras.layers.Flatten(name='flatten')(embedding_cat)

# embedding the sku
inputs_sku = tf.keras.layers.Input(shape=(1,),name = 'in_sku')
embedding_sku = tf.keras.layers.Embedding(input_dim=PRICING['sku'].nunique(), output_dim=100, input_length=1,name = 'embedding_sku')(inputs_sku)
embedding_flat_sku = tf.keras.layers.Flatten(name='flatten2')(embedding_sku)

#### Defining Functions to Use to build and tune model
## Hidden Layers
def create_hidden(nodes_list, activation_function = 'elu'):
    '''
    creates the hidden layers for the model
    nodes_list length indicates the number of layers created
    nodes_list values indicate the number of hidden nodes per layer (in order)
    activation_function is a string that indicates the activation function to be used; default elu
    '''
    # Initialize first hidden node
    hidden = tf.keras.layers.Dense(nodes_list[1],activation = activation_function)(inputs_concat2)
    # loop through remaining hidden layers
    if len(nodes_list) > 1:
        for i in range(len(nodes_list)-1):
            hidden = tf.keras.layers.Dense(nodes_list[i], activation = activation_function)(hidden)
    return hidden

## Concatenation of all input layers
# combining the categorical embedding layers
cats_concat = tf.keras.layers.Concatenate(name = 'concatenation1')([embedding_flat_cat, embedding_flat_sku])
#input for the quantity, price,order, and duration
inputs_num = tf.keras.layers.Input(shape=(3,),name = 'in_num')
#combinging the all input layers
inputs_concat2 = tf.keras.layers.Concatenate(name = 'concatenation')([cats_concat, inputs_num])

hidden = create_hidden(nodes_list = [20,10,11], activation_function = 'elu')

## Output layer
outputs = tf.keras.layers.Dense(1, name = 'out')(hidden)

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





