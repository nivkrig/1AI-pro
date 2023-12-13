import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from keras.datasets import mnist
print('water')
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


plt.imshow(x_train[3], interpolation='nearest')
plt.gray()
plt.show()

