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

# A utility method to create a tf.data dataset from a Pandas Dataframe
# def df_to_dataset(PRICING, shuffle=True, batch_size=32):
#   dataframe = dataframe.copy()
#   labels = dataframe.pop('target')
#   ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
#   if shuffle:
#     ds = ds.shuffle(buffer_size=len(dataframe))
#   ds = ds.batch(batch_size)
#   return ds

### First step is to encode the categorical variables: category and SKU
category = feature_column.categorical_column_with_vocabulary_list(
      'category', PRICING.category.unique())

category_embedding = feature_column.embedding_column(category, dimension=1)

sku = feature_column.categorical_column_with_vocabulary_list(
      'sku', PRICING.sku.unique())

sku_embedding = feature_column.embedding_column(sku, dimension=1)


price= feature_column.numeric_column('price')
duration = feature_column.numeric_column('duration')

## creating feature columns
feature_columns=[sku_embedding,price,duration, category_embedding]


## creating feature dense layer
feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

# model = tf.keras.Sequential([
#   feature_layer,
#   layers.Dense(128, activation='relu'),
#   layers.Dense(128, activation='relu'),
#   layers.Dropout(.1),
#   layers.Dense(1)
# ])

# model.compile(optimizer='adam',
#               loss=tf.keras.losses.BinaryCrossentropy,
#               metrics=['accuracy'])

# model.fit(train,
#           validation_data=val,
#           epochs=10)


