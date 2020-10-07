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
import cv2

image_path = '/home/togoldor/IA/img/car.jpg'

X = []
y = []

# convert color image to 2D array (grayscale) & rescale
data = cv2.imread(image_path,0) / 255.0
label = 0 # label/class of the image
X.append(data)
y.append(label)

# loop trough all images ...

# split for training & testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)