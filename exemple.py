import numpy as np
import cv2
from sklearn.model_selection import train_test_split

X = []
y = []
image_path = '/home/togoldor/car-number/img/zero.jpg'
print(image_path)
data = cv2.imread(image_path,0) / 255.0
label = 0
X.append(data)
y.append(label)
print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)