import tensorflow.keras as keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.models import Sequential
from PIL import Image
mnist = tf.keras.datasets.mnist
import cv2


mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)
print(x_test)
new_model = tf.keras.models.load_model('epic_num_reader.model')
predictions = new_model.predict(x_test)

print(np.argmax(predictions[0]))
print(np.argmax(predictions[0]))