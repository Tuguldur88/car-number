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
from sklearn.model_selection import train_test_split

image_path = '/home/togoldor/car-number/img/car.jpg'
model = Sequential()

y = []

image = tf.keras.preprocessing.image.load_img(image_path)
input_arr = keras.preprocessing.image.img_to_array(image)
input_arr = np.array([input_arr])  # Convert single image to a batch.
predictions = model.predict(input_arr)
print(predictions)


label = 0 # label/class of the image
y.append(label)

X_train, X_test, y_train, y_test = train_test_split(predictions, y)