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


# model = Sequential()
image_path = '/home/togoldor/car-number/img/car.jpg'


image = tf.keras.preprocessing.image.load_img(image_path)
# input_arr = keras.preprocessing.image.img_to_array(image)
# input_arr = np.array([input_arr])  # Convert single image to a batch.
# predictions = model.predict(input_arr[None,:,:])
# print(input_arr)
# print("wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww")
# print(predictions)
# print("wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww")






mnist = tf.keras.datasets.mnist
print(mnist.load_data())
print("wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww")
(x_train, y_train),(x_test, y_test) = mnist.load_data()
# print(x_train[0])
plt.imshow(x_train[0],cmap=plt.cm.binary)
# plt.show()
# print(y_train[0])

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# print(x_train[0])

plt.imshow(x_train[0],cmap=plt.cm.binary)
# plt.show()

# model = tf.keras.models.Sequential()

# model.add(tf.keras.layers.Flatten())

# model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

# model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

# model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# model.fit(x_train, y_train, epochs=3)

# val_loss, val_acc = model.evaluate(x_test, y_test)
# # # print(val_loss)
# # # print(val_acc)

# model.save('epic_num_reader.model')
new_model = tf.keras.models.load_model('epic_num_reader.model')
print(new_model)
print(x_test)
predictions = new_model.predict(x_test)
# print(predictions)

print(np.argmax(predictions[0]))
print(np.argmax(predictions[0]))
print(np.argmax(predictions[0]))
print(np.argmax(predictions[0]))