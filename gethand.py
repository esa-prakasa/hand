import cv2
import numpy as np
from time import perf_counter 
import os

#img = cv2.imread("C:\\Users\\INKOM06\\Documents\\a1OpenCVcodes\\data\\esa.jpg",1)



def getSmallerSize(img):
	scRatio = 0.9
	newRow = int(img.shape[1]*scRatio)
	newCol = int(img.shape[0]*scRatio)
	img = cv2.resize(img,(newRow, newCol) , interpolation = cv2.INTER_AREA)
	return img
#img = getSmallerSize(img)


#face_cascade = cv2.CascadeClassifier("C:\\Users\\Esa\\Documents\\a1OpenCVcodes\\data\\haarcascade_frontalface_default.xml");
#face_cascade = cv2.CascadeClassifier("C:\\Users\\INKOM06\\Documents\\a1ocvCodes\\data\\haarcascade_frontalface_default.xml");



face_cascade = cv2.CascadeClassifier("C:\\Users\\INKOM06\\Documents\\a1ocvCodes\\hand\\aGest.xml");
#C:\Users\INKOM06\Documents\a1ocvCodes\data


cap = cv2.VideoCapture(0)
t1 = perf_counter() 
oldDt = 0
timeInt = 3

ret, frame0 = cap.read()
M0 = frame0.shape[0]
N0 = frame0.shape[1]
ratio = 0.1


while(True):
	ret, frame = cap.read()
	frameOri = frame
	frame = cv2.resize(frame, (int(ratio*N0),int(ratio*M0)))
	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray,1.08,3)
	faceDetected = False

	t2 = perf_counter()
	dt = int(t2-t1)
	newDt = dt

	for (x,y,w,h) in faces:
		faceDetected = True
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
		#samp = frameOri[int(x):int(x+w), int(y):int(y+h), :]		
		samp = frameOri[int(y):int(y+h), int(x):int(x+w),  :]		
		cv2.imshow('Detected faces',frame)
		#if ((dt%timeInt) == 0):
		#	cv2.imwrite("C:\\Users\\INKOM06\\Documents\\a1OpenCVcodes\\data\\capture\\samp_"+str(dt)+".jpg",samp)

	#if (newDt != oldDt):
	#	oldDt = newDt
	#	os.system('cls')
	#	print("Time: %d" % dt)




    
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break





cv2.waitKey(0)
cv2.destroyAllWindows()
