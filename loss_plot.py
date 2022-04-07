import Project2_Function
import matplotlib.pyplot as plt

#loss plot  
model_history = model.fit(x=input_dict, y=train['quantity'], batch_size=grid_row['batch_size'], epochs=grid_row['epochs'])
plt.plot(model_history.history['loss'],label='model1')
plt.legend()
plt.ylabel("loss")
plt.xlabel("epoch")
plt.show()

#planning on cutting down number of model to try then stacking the final X amount on the 
#same plot. 7,000 + lines on one plot is not a good idea.