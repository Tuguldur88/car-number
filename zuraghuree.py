import cv2
import numpy as np

img = cv2.imread("/home/togoldor/car-number/img/car.jpg")

grises= cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

bordes= cv2.Canny(grises, 100, 250)

ctns = cv2.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
ctns = ctns[0] if len(ctns)==2 else ctns[1]
for c in ctns:
    cv2.drawContours(img,[c], -1,(0,0,255),2)

print ('Numero de contornos es ', len(ctns))
texto= 'Contornos encontrados ' + str(len(ctns))

cv2.putText(img, texto, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
    (255, 0, 0), 1)


cv2.imshow('Bordes', bordes)
cv2.waitKey(0)
cv2.destroyAllWindows()