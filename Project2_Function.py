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
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import feature_column
from tensorflow.keras import layers
from itertools import product
import csv
import time


#### Reading and Cleaning
train = pd.read_csv('train65.csv')
val = pd.read_csv('val65.csv')
test= pd.read_csv('test65.csv')
train = train.iloc[: , 1:]
val = val.iloc[: , 1:]
test = test.iloc[: , 1:]

# Check if the values are consecutively encoded:
def checkConsecutive(l):
    return (sorted(l) == list(range(min(l), max(l) + 1)))
print(checkConsecutive(np.unique(pd.concat([train, val, test])['category'])))
print(checkConsecutive(np.unique(pd.concat([train, val, test])['sku'])))

n_unique_cats = len(np.unique(pd.concat([train, val, test])['category']))
n_unique_skus = len(np.unique(pd.concat([train, val, test])['sku']))


#### Defining Functions to Use to build and tune model
## kernel_initializer
def get_kernel_initializer(activation_function, initializer_name):
    '''
    :param initializer_name:
        - tanh options: 'glorot_uniform','glorot_normal'
        - sigmoid options: 'uniform', 'untruncated_normal'
        - relu and friends options: 'he_normal', 'he_uniform', 'he_avg_normal', 'he_avg_uniform'
    :param activation_function:
        - options: 'tanh', 'sigmoid', 'elu','relu','prelu', 'leaky relu'
    :return: a string or a keras object for kernel_initializer parameter
    '''
    # No activation function was called, so none is returned
    if initializer_name is None :
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
def create_hidden(inputs, nodes_list, activation_function, batch_norm = False, initializer_name = None):
    '''
    creates the hidden layers for the model

    :param inputs: input layer for first hidden node
    :param nodes_list: number of nodes per hidden layer
    :param activation_function: string that indicates the activation function to be used
    :param batch_norm: either True or False indicating whether or not to perform Batch Normalization (only does before activation)
    :param initializer_name:
    '''
    # Initialize first hidden node
    kernel_initializer = get_kernel_initializer(activation_function, initializer_name)

    if batch_norm:
        hidden = tf.keras.layers.Dense(nodes_list[0], kernel_initializer=kernel_initializer)(inputs)
        BN = tf.keras.layers.BatchNormalization()(hidden)
        hiddenAct = tf.keras.layers.Activation('elu')(BN)
        if len(nodes_list) > 1:
            for i in range(len(nodes_list) - 1):
                hidden = tf.keras.layers.Dense(nodes_list[i+1], kernel_initializer=kernel_initializer)(hiddenAct)
                BN = tf.keras.layers.BatchNormalization()(hidden)
                hiddenAct = tf.keras.layers.Activation('elu')(BN)
        return hiddenAct
    else:
        hidden = tf.keras.layers.Dense(nodes_list[0], kernel_initializer=kernel_initializer)(inputs)
        if len(nodes_list) > 1:
            for i in range(len(nodes_list) - 1):
                hidden = tf.keras.layers.Dense(nodes_list[i+1], kernel_initializer=kernel_initializer)(hidden)
        return hidden


## Optimizers
def get_optimizer(learning_rate, optimizer_name = None, clipnorm = False):
    '''
    :param learning_rate: learning rate
    :param optimizer_name: 'momentum','nesterov','RMSprop','Adam', 'learning rate scheduling', 'plain SGD'
    :return: optimizer arg for model.compile
    '''
    if clipnorm:
        clip = 1
    else:
        clip = None
    if optimizer_name == 'momentum':
        return tf.keras.optimizers.SGD(learning_rate = learning_rate, momentum = 0.9, clipnorm = clip)
    elif optimizer_name == 'nesterov':
        return tf.keras.optimizers.SGD(learning_rate = learning_rate, momentum = 0.9, nesterov = True, clipnorm = clip)
    elif optimizer_name == 'adagrad':
        return tf.keras.optimizers.Adagrad(learning_rate = learning_rate, initial_accumulator_value = 0.1, epsilon = 1e-07, clipnorm = clip)
    elif optimizer_name == 'RMSprop':
        return tf.keras.optimizers.RMSprop(learning_rate = learning_rate, rho = 0.9, momentum = 0.0, epsilon = 1e-07, clipnorm = clip)
    elif optimizer_name == 'Adam':
        return tf.keras.optimizers.Adam(learning_rate = learning_rate, beta_1 = 0.9, beta_2 = 0.99, epsilon = 1e-07, clipnorm = clip)
    elif optimizer_name == 'learning rate scheduling':
        return tf.keras.optimizers.schedules.ExponentialDecay(learning_rate, 10000, 0.95, clipnorm = clip)
    elif optimizer_name == 'plain SGD':
        return tf.keras.optimizers.SGD(learning_rate = learning_rate, clipnorm = clip)


## Creating model based on inputs
def create_model(nodes_list, activation_function, batch_norm = False,
                 initializer_name = None):
    '''
    :param nodes_list:
    :param activation_function:
    :param batch_norm:
    :param initializer_name:
    :return: tf.keras model
    '''
    #### Embedding and Creating Layers
    ## First step is to encode the categorical variables: category and SKU
    # category
    output_cat = 16
    output_sku = 50
    # nodes_list[0]*2 - output_cat - 3

    tf.keras.backend.clear_session()
    inputs_cat = tf.keras.layers.Input(shape=(1,),name = 'in_cats')
    embedding_cat = tf.keras.layers.Embedding(input_dim=n_unique_cats, output_dim=output_cat, input_length=1,name = 'embedding_cat')(inputs_cat)
    embedding_flat_cat = tf.keras.layers.Flatten(name='flatten')(embedding_cat)

    # sku
    inputs_sku = tf.keras.layers.Input(shape=(1,),name = 'in_sku')
    embedding_sku = tf.keras.layers.Embedding(input_dim=n_unique_skus, output_dim=output_sku, input_length=1,name = 'embedding_sku')(inputs_sku)
    embedding_flat_sku = tf.keras.layers.Flatten(name='flatten2')(embedding_sku)

    ## Concatenation of all input layers
    # combining the categorical embedding layers
    cats_concat = tf.keras.layers.Concatenate(name = 'concatenation1')([embedding_flat_cat, embedding_flat_sku])
    #input for the quantity, price,order, and duration
    inputs_num = tf.keras.layers.Input(shape=(3,),name = 'in_num')
    #combinging the all input layers
    inputs_concat2 = tf.keras.layers.Concatenate(name = 'concatenation')([cats_concat, inputs_num])

    ## Defining Hidden Layers
    hidden = create_hidden(inputs_concat2, nodes_list = nodes_list, activation_function = activation_function,
                           batch_norm = batch_norm, initializer_name = initializer_name)

    ## Output layer/ Finalize Inputs
    outputs = tf.keras.layers.Dense(1, name = 'out')(hidden)
    inputs=[inputs_cat,inputs_sku,inputs_num]

    #### Create Model
    model = tf.keras.Model(inputs = inputs, outputs = outputs)

    return model


#### Training Model
def expand_grid(dictionary):
   return pd.DataFrame([row for row in product(*dictionary.values())],
                       columns=dictionary.keys())

# Creating a a grid with combinations we might like to try
dictionary = {'nodes_list': [[1000, 500, 250, 125, 75, 25], [5000, 2500, 1250, 750, 250, 100, 50]],
              'activation_function': ["sigmoid","tanh","relu","elu"],
              'learning_rate': [0.001, 0.01,0.1],
              'batch_norm': [True, False],
              'initializer_name': ['glorot_uniform', 'glorot_normal', 'uniform', 'untruncated_normal', 'he_normal', 'he_uniform', 'he_avg_normal', 'he_avg_uniform'],
              'optimizer_name':["plain SGD","nesterov","RMSprop","Adam"],
              'epochs':[100],
              'batch_size': [28],
              'clipnorm': [False]} # prioritize 28

grid = expand_grid(dictionary)

# Remove incompatible combinations for weight initialization and activaiton functions
grid = grid[-((grid['activation_function'] != 'tanh') & (grid['initializer_name'] == 'glorot_uniform'))]
grid = grid[-((grid['activation_function'] != 'tanh') & (grid['initializer_name'] == 'glorot_normal'))]
grid = grid[-((grid['activation_function'] != 'sigmoid') & (grid['initializer_name'] == 'uniform'))]
grid = grid[-((grid['activation_function'] != 'sigmoid') & (grid['initializer_name'] == 'untruncated_normal'))]
grid = grid[-((grid['activation_function'] == 'tanh') & (grid['initializer_name'] == 'he_normal'))]
grid = grid[-((grid['activation_function'] == 'tanh') & (grid['initializer_name'] == 'he_uniform'))]
grid = grid[-((grid['activation_function'] == 'tanh') & (grid['initializer_name'] == 'he_avg_normal'))]
grid = grid[-((grid['activation_function'] == 'tanh') & (grid['initializer_name'] == 'he_avg_uniform'))]
grid = grid[-((grid['activation_function'] == 'sigmoid') & (grid['initializer_name'] == 'he_normal'))]
grid = grid[-((grid['activation_function'] == 'sigmoid') & (grid['initializer_name'] == 'he_uniform'))]
grid = grid[-((grid['activation_function'] == 'sigmoid') & (grid['initializer_name'] == 'he_avg_normal'))]
grid = grid[-((grid['activation_function'] == 'sigmoid') & (grid['initializer_name'] == 'he_avg_uniform'))]

grid = grid[-((grid['batch_size'] == 1) & (grid['batch_norm'] == True))]

grid = grid.reset_index(drop = True)


### If we want to do some sort of loop we can do it with these 6 lines and add some lines to save the information we want:
## splitting the data into test and train sets
# train, test = train_test_split(PRICING, test_size=0.2)
# train, val = train_test_split(train, test_size=0.2)

def get_input_dict(data):
    ## seperating the numerical features from rest of dataset
    num_features=data.drop(['sku'], axis=1)
    num_features=num_features.drop(['category'], axis=1)
    num_features=num_features.drop(['quantity'], axis=1)

    ## creates an input dictionary for the model
    input_dict= {
        'in_cats':data["category"],
        "in_sku":data["sku"],
        "in_num": num_features
    }
    return input_dict

## Intuitive Selection
# input_dict_train = get_input_dict(train)
# input_dict_val = get_input_dict(val)


# model = create_model(nodes_list = [1000, 500, 250, 125, 75, 25], activation_function='elu', batch_norm = True,
#                      initializer_name = 'he_avg_uniform')
# optimizer = get_optimizer(0.01, 'Adam')
# model.compile(loss='mse', optimizer=optimizer)

# import time
# start = time.time()
# model_history = model.fit(x=input_dict_train, y=train['quantity'], batch_size=28, epochs=100,
#                           validation_data = (input_dict_val,val['quantity']))
# total_time = time.time()-start
# print(total_time)

# model.summary()

# model.summary()
# model.save('models/' + 'Original_M.h5')



# how many random models to try and save
n_random = 1
random_rows = [random.randint(0, len(grid) - 1) for i in range(n_random)]
histories = []
# Run and fit the randomly selected models
input_dict_train = get_input_dict(train)
input_dict_val = get_input_dict(val)
for i in random_rows:
    model_name = 'model_' + str(i)
    print('running', model_name)
    grid_row = grid.loc[i]
    print('\n', grid_row)
    model = create_model(grid_row['nodes_list'], grid_row['activation_function'], batch_norm = grid_row['batch_norm'],
                         initializer_name = grid_row['initializer_name'])

    optimizer = get_optimizer(grid_row['learning_rate'], grid_row['optimizer_name'])
    model.compile(loss='mse', optimizer=optimizer)

    # count hidden layers in model
    hidden_layers = len(grid_row['nodes_list'])

# saving the best weights for the selected model
    checkpoint_cb=tf.keras.callbacks.ModelCheckpoint(
    filepath='models/' + str(model_name) + '_1.h5',
    save_freq=  'epoch',
    save_best_only=True)

# stopping the training if the validation loss does not improve for 5 epochs
    early_stopping_cb=tf.keras.callbacks.EarlyStopping(patience=5,restore_best_weights=True)

## terminate on NA
    terminate_na = tf.keras.callbacks.TerminateOnNaN()

# might need to fix batch size to a higher amount if the training is taking too long
    start = time.time()
    model_history = model.fit(x=input_dict_train, y=train['quantity'], batch_size=grid_row['batch_size'],
                              epochs=grid_row['epochs'], validation_data = (input_dict_val, val['quantity']),
                              callbacks=[checkpoint_cb,early_stopping_cb, terminate_na])
    histories.append(model_history)
    total_time = time.time()-start
    history=model_history.history

    # read header from models.csv file
    with open('models.csv', "r") as f:
        reader = csv.reader(f)
        for header in reader:
            break
    #rename header column 'model' due to reoccuring text error
    header[0]='model'

    # add row to CSV file for each model ran in loop
    with open('models.csv', "a", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)

        writer.writerow({'train_loss':history["loss"],'hidden_layers':hidden_layers,'train_time':total_time,'model':model_name,
                         'nodes_list':grid_row['nodes_list'],'activation_function':grid_row['activation_function'],
                         'batch_norm':grid_row['batch_norm'] ,'initializer_name':grid_row['initializer_name'],
                         'learning_rate':grid_row['learning_rate'],'optimizer_name':grid_row['optimizer_name'],
                         'batch_size':grid_row['batch_size'],'clipnorm': grid_row['clipnorm'],
                         'val_loss':history["val_loss"],'epochs':len(history['val_loss'])}) # This is the amount of epochs that ran, including after the ideal model. 



    # Save results
    #model.save('models/' + str(model_name) + '_1.h5')

# Print Model Results
# for i in range(len(histories)):
#     print('model_' + str(random_rows[i]) + ':\n' +
#           str(grid.loc[random_rows[i]]) + '\n' +
#           str(histories[i].history))




#### NOTES#######
# Because of early stopping, no need to try out different epoch sizes. 
# An example is that setting the epochs to 10, early stopping might stop the training after 5 epochs.


# Training is extremely slow with a small batch size


# The best model is the one with the lowest validation loss. Need to compare validation loss between models. 
# If a certain activation function has higher validation loss than the minumum after a couple tries, drop that activation function from the grid
