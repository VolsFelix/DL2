import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import feature_column
from tensorflow.keras import layers
import os
import matplotlib.pyplot as plt
import time
print(tf.__version__)
import keras_tuner as kt
#def get_dataset(file_path):
import tensorboard


#numeric features being fed into the model:
feature_columns = []
feature_columns.append(tf.feature_column.numeric_column('price'))
feature_columns.append(tf.feature_column.numeric_column('order'))
feature_columns.append(tf.feature_column.numeric_column('duration'))


# categorical columns using the lists created above:
sku_list = list(range(64902)) #full  dataset
cat_list = list(range(32)) #full  dataset
sku_col = tf.feature_column.categorical_column_with_vocabulary_list(
'sku', sku_list)
cat_col = tf.feature_column.categorical_column_with_vocabulary_list(
'category', cat_list)
sku_emb = tf.feature_column.embedding_column(sku_col,dimension=50) #output?, dimension?
cat_emb = tf.feature_column.embedding_column(cat_col,dimension=16) #output?, dimension?
feature_columns.append(sku_emb)
feature_columns.append(cat_emb)                      

feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

[]
min = [85,45]
#[50,35,20]
feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
model = tf.keras.models.Sequential()
model.add(feature_layer)
he_avg_init = tf.keras.initializers.VarianceScaling(scale = 2., mode ="fan_avg", distribution = 'uniform')    
for i in min:
    model.add(layers.Dense(units=min[i],activation='ELU'))
    #,kernel_initializer =he_avg_init
    #model.add(tf.keras.layers.BatchNormalization())

model.add(layers.Dense(1))
#compile model
hp_learning_rate =  0.0005
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
            loss='mse', metrics = 'mse')
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3,restore_best_weights=True)
start = time.time()

model.fit(train_data, validation_data =val_data,  epochs=15, callbacks = [stop_early])

time.time()-start

os.chdir('/Users/katemac2021/Desktop/MLProject2/data')

def get_dataset(file_path):
  dataset = tf.data.experimental.make_csv_dataset(
      file_path,
      batch_size=50, 
      label_name='quantity',
      na_value="?",
      num_epochs=1,
      ignore_errors=True)
  return dataset

raw_train_data = get_dataset("train65.csv") 
train_data = raw_train_data.shuffle(500)
train_data
val_data = get_dataset("val65.csv")
test_data = get_dataset("test65.csv")



import keras_tuner as kt


#trial : big
def model_builder(hp):
    min = [60,30,10,1,1,1,1,1,1,1]
    max = [100,80,70,60,50,40,30,20,15,10]
    feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
    model = tf.keras.models.Sequential()
    model.add(feature_layer)
    for i in range(hp.Int('num_layers', 2, 10)):
        model.add(layers.Dense(units=hp.Int('units_' + str(i),
                                            min_value=min[i],
                                            max_value=max[i], 
                                            step=5),
                               activation=hp.Choice("activation", 
                                ['sigmoid', 'tanh', 'elu',"relu", "tanh"])))
        model.add(tf.keras.layers.BatchNormalization())
    model.add(layers.Dense(1))
    #compile model
    hp_lr =  hp.Choice('learning_rate', values=[1e-2,5e-3, 1e-3, 5e-4,1e-4])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_lr),
                loss='mse', metrics = 'mse')
    return model

#trial : small
def model_builder2(hp):
    min = [30,15,1,1,1,1,1,1,1,1]
    max = [50,40,30,20,20,20,10,10,10,5]
    feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
    model = tf.keras.models.Sequential()
    model.add(feature_layer)
    for i in range(hp.Int('num_layers', 2, 10)):
        model.add(layers.Dense(units=hp.Int('units_' + str(i),
                                            min_value=min[i],
                                            max_value=max[i], #100,90,80,70,60,50,40,
                                            step=5),
                               activation=hp.Choice("activation", ['elu',"relu"])))
    model.add(layers.Dense(1))
    #compile model
    hp_learning_rate =  hp.Choice('learning_rate', values=[1e-2,5e-3, 1e-3, 5e-4,1e-4])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                loss='mse', metrics = 'mse')
    return model

def model_builder_relu(hp):
    min = [30,15,1,1,1,1,1,1,1,1]
    max = [50,40,30,20,20,20,10,10,10,5]
    feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
    model = tf.keras.models.Sequential()
    model.add(feature_layer)
    he_avg_uni = tf.keras.initializers.VarianceScaling(scale = 2., mode ="fan_avg", distribution = 'uniform')
    he_avg_normal = tf.keras.initializers.VarianceScaling(scale = 2., mode ="fan_avg", distribution = 'normal')
    for i in range(hp.Int('num_layers', 2, 10)):
        model.add(layers.Dense(units=hp.Int('units_' + str(i),
                                            min_value=min[i],
                                            max_value=max[i], #100,90,80,70,60,50,40,
                                            step=5),
                                            activation=hp.Choice('activation', values = ['elu', 'relu']),
                                            kernel_initializer = hp.Choice('initializer', values = [he_avg_uni, he_avg_normal, 'he_normal', 'he_uniform'])))
        model.add(tf.keras.layers.BatchNormalization())
    model.add(layers.Dense(1))
    #compile model
    hp_learning_rate =  hp.Choice('learning_rate', values=[5e-3, 1e-3, 5e-4])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                loss='mse', metrics = 'mse')
    return model

#First tuner_hb
tuner_hb = kt.Hyperband(model_builder2,
                     objective='val_mse',
                     max_epochs=8,
                     factor=3,
                     #max_trials=20,
                     seed=42,
		             directory="output_hb",
		             project_name="Hyperband")
#objective val_loss
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3,restore_best_weights=True)
tuner_hb.search_space_summary()
tuner_hb.search(train_data, epochs=8, validation_data=val_data, callbacks=[stop_early])
tuner_hb.results_summary()
modelhp_April10 = tuner_hb.get_best_models(num_models=1)
modelhp_April10.save_model("modelhp_April10")
modelhp_April10.save('models/' + 'hyperband_April10')



#Second tuner_hb

tuner_hb2 = kt.Hyperband(model_builder2,
                     objective='val_mse',
                     max_epochs=8,
                     factor=3, 
                     #max_trials=20,
                     seed=42,
		             directory="output_hb2",
		             project_name="Hyperband2")
#objective val_loss
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3,restore_best_weights=True)
tuner_hb2.search_space_summary()
tuner_hb2.search(train_data, epochs=8, validation_data=val_data, callbacks=[stop_early])
tuner_hb2.results_summary()
modelhp_April10 = tuner_hb.get_best_models(num_models=1)
modelhp_April10.save_model("modelhp_April10")
modelhp_April10.save('models/' + 'hyperband_April10')

#checkpoint
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath="data",
    monitor=['val_mse', 'train_mse'],
    mode='max')

besthp = tuner_hb.get_best_hyperparameters(num_trials=1)[0]
best_model_hb2 = tuner_hb2.get_best_models(num_models=1)[0]

best_model_hb2.evaluate(val_data)
model = tuner_hb.hypermodel.build(besthp)
best_model.summary()
best_model_hb2.save('models/' + 'hyperband2_50_April11')
model.evaluate(val_data)
#7:48
tuner_bo = kt.BayesianOptimization(
            model_builder,
            objective='val_mse',
            max_trials=20,
            seed=42,
            directory="output1",
		    project_name="BayesianOpti1"
        )
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3,restore_best_weights=True)
tuner_bo.search_space_summary()

tuner_bo.search(train_data, epochs=6, validation_data=val_data, callbacks=[stop_early,tensorboard_callback])
tuner_bo.results_summary(num_trials=10)
tuner_bo.get_best_models(num_models=1)
tuner_bo.get_best_hyperparameters(num_trials=1)
best_model_bo = tuner_bo.get_best_models(num_models=1)



import keras_tuner as kt
tuner_rs = kt.RandomSearch(
            model_builder2,
            objective='val_mse',
            seed=42,
            max_trials=20,
            directory="output2",
		    project_name="RS")
tuner_rs.search_space_summary()
#log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
tuner_rs.search(train_data, epochs=8, validation_data=val_data, callbacks=[stop_early,tf.keras.callbacks.TensorBoard("rs/logs")])
tuner_rs.results_summary()
best_model_rs_1044 = tuner_rs.get_best_models(num_models=1)[0]
best_model_rs_1044.evaluate(val_data)
best_model_rs_1044.save("best_model_rs_1044")
model =best_model_rs_1044
 model.get_config()
 model.optimizer.get_config()
for layer in model.layers[1:]:
    print(layer.output_shape)
import tensorboard

model = tf.keras.models.load_model("models/hyperband_1") #1119
model = tf.keras.models.load_model("models/hyperband_April10") #1071
model = tf.keras.models.load_model("models/hyperband2_50_April11") #1050.50
model = tf.keras.models.load_model("models/rs_April12") #1044.79

weights = [glorot_uniform(seed=random.randint(0, 1000))(w.shape) if w.ndim > 1 else w for w in model.get_weights()]
