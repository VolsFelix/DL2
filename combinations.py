import pandas as pd
from itertools import product

def expand_grid(dictionary):
   return pd.DataFrame([row for row in product(*dictionary.values())], 
                       columns=dictionary.keys())

dictionary = {'batch_size': [100], 
              'num_hidden_layers': [1,2,3,4,5,6,7], 
              'num_hidden_neurons': [50,100,250],
              'act_func': ["sigmoid","tanh","relu","leaky relu","prelu","elu"],
              'optimizers':["plain SGD","momentum","nesterov","adagrad","rmsprop","adam","learning rate scheduling"]}

expand_grid(dictionary)

#can reduce combinations by cutting out smaller number of layers like 1 & 2 
#since our data will require more layers than less to fully learn model

#values in num_hidden_neurons was arbitrarily made because wasn't sure 
#the best way to calculate the number. Found some stuff on the internet 
#that said it should be between the size of the input and output layer.
#It should be 2/3 the size of input layer plus the output layer