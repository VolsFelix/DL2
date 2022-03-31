#### Group 2 Deep Learning Project
# The goal of this assignment is to gather experience on the sensitivity of the algorithm to different
# kinds of tuning parameters: batch size, number of hidden layers, number hidden neurons, hidden
# activation functions (sigmoid, tanh, relu, leaky relu, prelu, elu),
# optimizers (plain SGD, momentum, nesterov, adagrad, rmsprop, adam, learning rate scheduling)

# The goal is to predict quantity sold of a given product
# as accurately as possible by tuning the learning procedure
import keras.initializers.initializers_v2
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import feature_column
from tensorflow.keras import layers
import warnings

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



#### Defining Functions to Use to build and tune model
## kernel_initializer
def get_kernel_initializer(activation_function, initializer_name):
    '''
    :param initializer_name:
        tanh options: 'glorot_uniform','glorot_normal'
        sigmoid options: 'uniform', 'untruncated_normal'
        relu and friends options: 'he_normal', 'he_uniform', 'he_avg_normal', 'he_avg_uniform'
    :param activation_function:
        options: 'tanh', 'sigmoid', 'elu','relu','prelu', 'leaky relu'
    :return: either a string or a keras object for kernel_initializer parameter
    '''
    # No activation function was called, so none is returned
    if activation_function is None:
        return None

    # tanh activation function weight initializers
    if (activation_function == 'tanh') & (initializer_name in ['glorot_uniform','glorot_normal']):
        return initializer_name

    # sigmoid activation function weight initializers
    elif (activation_function == 'sigmoid') & (initializer_name in ['uniform', 'untruncated_normal']):
        return keras.initializers.VarianceScaling(scale = 16., mode = 'fan_avg', distribution = initializer_name)

    # relu and friends activaiton function weight initializerss
    elif activation_function in ['elu','relu','prelu', 'leaky relu']:
        if initializer_name in ['he_normal', 'he_uniform']:
            return initializer_name
        elif initializer_name == 'he_avg_normal':
            return keras.initializers.VarianceScaling(scale = 2., mode = 'fan_avg', distribution = 'normal')
        elif initializer_name == 'he_avg_uniform':
            return keras.initializers.VarianceScaling(scale = 2., mode = 'fan_avg', distribution = 'uniform')
        else:
            print('Not a valid combination of initializers and activation functions;\n'
                  'No weight initializer will be used')
            return None

    # If given a bad combination or an incorrect activation function -- give warning
    else:
        warnings.warn('\n\nNot a valid combination of activation and initializers;\n'
                      'Or not a valid activation function entry;\n'
                      'No weight initializer will be used\n')
        return None



## Hidden Layers
def create_hidden(nodes_list, activation_function, batch_norm = False, initializer_name = None):
    '''
    creates the hidden layers for the model
    :param nodes_list: length indicates the number of layers created;
     values indicate the number of hidden nodes per layer (in order)
    :param activation_function: string that indicates the activation function to be used
         options: 'tanh', 'sigmoid', 'elu','relu','prelu', 'leaky relu'
    :param batch_norm: either True or False indicating whether or not to perform Batch Normalization (only does before activation)
    :param initializer_name:
        tanh options: 'glorot_uniform','glorot_normal'
        sigmoid options: 'uniform', 'untruncated_normal'
        relu and friends options: 'he_normal', 'he_uniform', 'he_avg_normal', 'he_avg_uniform'
    '''
    # Initialize first hidden node
    kernel_initializer = get_kernel_initializer(activation_function, initializer_name)

    if batch_norm:
        hidden = tf.keras.layers.Dense(nodes_list[1], kernel_initializer=kernel_initializer)(inputs_concat2)
        BN = tf.keras.layers.BatchNormalization()(hidden)
        hiddenAct = tf.keras.layers.Activation('elu')(BN)
        if len(nodes_list) > 1:
            for i in range(len(nodes_list) - 1):
                hidden = tf.keras.layers.Dense(nodes_list[i], kernel_initializer=kernel_initializer)(hiddenAct)
                BN = tf.keras.layers.BatchNormalization()(hidden)
                hiddenAct = tf.keras.layers.Activation('elu')(BN)
        return hiddenAct
    else:
        hidden = tf.keras.layers.Dense(nodes_list[1], kernel_initializer=kernel_initializer)(inputs_concat2)
        if len(nodes_list) > 1:
            for i in range(len(nodes_list) - 1):
                hidden = tf.keras.layers.Dense(nodes_list[i], kernel_initializer=kernel_initializer)(hidden)
        return hidden

## First step is to encode the categorical variables: category and SKU
# embedding category
inputs_cat = tf.keras.layers.Input(shape=(1,),name = 'in_cats')
embedding_cat = tf.keras.layers.Embedding(input_dim=PRICING['category'].nunique()+1, output_dim=16, input_length=1,name = 'embedding_cat')(inputs_cat)
embedding_flat_cat = tf.keras.layers.Flatten(name='flatten')(embedding_cat)

# embedding the sku
inputs_sku = tf.keras.layers.Input(shape=(1,),name = 'in_sku')
embedding_sku = tf.keras.layers.Embedding(input_dim=PRICING['sku'].nunique(), output_dim=100, input_length=1,name = 'embedding_sku')(inputs_sku)
embedding_flat_sku = tf.keras.layers.Flatten(name='flatten2')(embedding_sku)

## Concatenation of all input layers
# combining the categorical embedding layers
cats_concat = tf.keras.layers.Concatenate(name = 'concatenation1')([embedding_flat_cat, embedding_flat_sku])
#input for the quantity, price,order, and duration
inputs_num = tf.keras.layers.Input(shape=(3,),name = 'in_num')
#combinging the all input layers
inputs_concat2 = tf.keras.layers.Concatenate(name = 'concatenation')([cats_concat, inputs_num])

hidden = create_hidden(nodes_list = [20,10,11], activation_function = 'elu', batch_norm = True, initializer_name = 'he_normal')

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





