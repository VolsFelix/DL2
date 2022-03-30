#Tuning Parameter Combinations 

expand.grid(batch_size = c("100"),
            num_hidden_layers = c("1","2","3","4","5","6"),
            num_hidden_neurons = c("50","100","200"), 
            act_func = c("sigmoid","tanh","relu","leaky relu","prelu","elu"), 
            optimizers = c("plain SGD","momentum","nesterov","adagrad","rmsprop","adam","learning rate scheduling")) 
        

#can reduce combinations by cutting out smaller number of layers like 1 & 2 
#since our data will require more layers than less to fully learn model

#values in num_hidden_neurons was arbitrarily made because wasn't sure 
#the best way to calculate the number. Found some stuff on the internet 
#that said it should be between the size of the input and output layer.
#It should be 2/3 the size of input layer plus the output layer
