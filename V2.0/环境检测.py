import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
#import pandas as pd

print("Tensorflow version:", tf.__version__)
print("Keras version:", keras.__version__)

model = keras.Sequential([
  keras.layers.Dense(10, input_shape=(3,)),
  keras.layers.Activation('relu'),
  keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.summary()
rewards = [1,2,3,4,5,6]
plt.plot(rewards)
plt.xlabel("Xlael")
plt.ylabel("Ylabel")
plt.title("function")
plt.show()