import numpy as np
import json
import h5py
import tensorflow as tf
import keras
from keras.datasets import mnist
from keras.models import model_from_json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0



model = tf.keras.models.load_model('model.keras')

example = x_train[3]
predict =  model.predict(example.reshape(1, 28, 28, 1))
predict = np.argmax(predict)
print(predict)