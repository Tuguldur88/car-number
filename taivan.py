import cv2
import numpy as np
import pytesseract
from PIL import Image , ImageFilter
import glob
import os ,errno
mydir = "/home/togoldor/car-number/data/training/training/0004.png"

for fil in glob.glob(mydir):
    image = cv2.imread(fil) 
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    ret,thresh_image = cv2.threshold(gray_image,80,255, cv2.THRESH_BINARY)
    print(thresh_image)
    cv2.imshow('sad',thresh_image)



print(pytesseract.image_to_string(thresh_image, lang="mon", config ='--oem 3 --psm 6'))

cv2.waitKey(0)