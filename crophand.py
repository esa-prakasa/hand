import cv2
import os
import numpy as np
import math


os.system("cls")

path = r"C:\Users\INKOM06\Pictures\_DATASET\handwash\s_cuci_tangan_11"
files = os.listdir(path)
fullPath = os.path.join(path,files[2])

img = cv2.imread(fullPath)

M = img.shape[0]
N = img.shape[1]

img2 = img[int(0.25*M):int(0.75*M), :, :]

img3 = cv2.resize(img,(200, 200) , interpolation = cv2.INTER_AREA) 

cv2.imshow("Crop", img2)
print("crop height: %d"%(img2.shape[0]))
print("crop width: %d"%(img2.shape[1]))


cv2.imshow("Resize", img3)


cv2.waitKey(0)
cv2.destroyAllWindows()
