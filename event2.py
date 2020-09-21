import numpy as np
import cv2
import os
import time
import pandas as pd 
from datetime import datetime
import math
import random


from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.callbacks import ModelCheckpoint

from keras.models import load_model
from keras.preprocessing import image
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels



def generateFileName():
	text = "ABCDEFGHIJKLMNOPQERSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
	fileName = ""
	for i in range(30):
		fileName = fileName+random.choice(text)
	return fileName



fileNameList = []

start_time = time.time()
saveFrames = False
saveOutVideo = False


checWindkSz = 200

def getDifferenceImage(gray, gray0):
	M = gray.shape[0]
	N = gray.shape[1]

	diffImage = np.zeros((M,N), dtype = np.int8)

	for i in range(M):
		for j in range(N):
			diffImage[i,j] = abs(gray[i,j] - gray0[i,j])

	return diffImage


os.system("cls")







kfold = "_fold1"

rootPath = "C:\\Users\\INKOM06\\Pictures\\handwash\\mod1\\trdataset\\"

modelPath = rootPath+kfold+"\\xmodel\\"


modelFileList = os.listdir(modelPath)
modIdx = 0
for modelFile in modelFileList:
  print(str(modIdx)+" "+modelFile)
  modIdx = modIdx + 1

#modIdx = int(input("Which model that will be used? "))
modIdx = 2


modelName = modelFileList[modIdx]
model = load_model(modelPath+modelName)

model.summary()

os.system("cls")






cap = cv2.VideoCapture(1)
gray0 = np.zeros((checWindkSz,checWindkSz), dtype = np.int8)


frameIdx = 0

mAvg =[0]*15
oldStat = "STANDBY"



while(True):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    gray = cv2.resize(gray, (checWindkSz,checWindkSz))


    cv2.imshow('frame',gray)
    deltaTime = time.time() - start_time
    
    deltaTimeInt = math.floor(deltaTime)



    diffImage = getDifferenceImage(gray, gray0)

    stdVal = np.std(diffImage)
    cv2.imshow('Difference Frame',diffImage)
    gray0 = gray


    position = (50,100)

# -------------- ORGINAL --------------
    # if (stdVal > 10):
    # 	infoStr = "REC"
    # 	colour = [0, 0, 255, 0]
    # 	fontSz = 1.5
    # 	fontTck = 4

    # if (stdVal < 10) :
    # 	infoStr = "StandBy"
    # 	colour = [0, 0, 0, 0]
    # 	fontSz = 0.8
    # 	fontTck = 2




    


    if (stdVal > 5):
    	mVal = 1
    	# recStat = True

    	# infoStr = "REC"
    	# colour = [0, 0, 255, 0]
    	# fontSz = 1.5
    	# fontTck = 4
    	# #fileName = generateFileName()
    	# #fileNameList.append(fileName)


 
    if (stdVal < 5):
    	mVal = 0
    	# recStat = False   

    	# infoStr = "StandBy"
    	# colour = [0, 0, 0, 0]
    	# fontSz = 0.8
    	# fontTck = 2


    mAvg.remove(mAvg[0])
    mAvg.append(mVal)
    sumAvg = sum(mAvg)

    if (sumAvg>3):
    	infoStr = "RECORD"
    	colour = [0, 0, 255, 0]
    	fontSz = 1
    	fontTck = 4
    else:
    	infoStr = "STANDBY"
    	colour = [0, 255, 0, 0]
    	fontSz = 1
    	fontTck = 4



    newStat = infoStr

    if (newStat=="RECORD") & (oldStat=="STANDBY"):
    	recordStatInfo = "Start to record"

    if (newStat=="STANDBY") & (oldStat=="RECORD"):
    	recordStatInfo = "End the recording"

    if (newStat=="RECORD") & (oldStat=="RECORD"):
    	recordStatInfo = "Still recording"

    if (newStat=="STANDBY") & (oldStat=="STANDBY"):
    	recordStatInfo = "Nothing to do"

    oldStat = newStat



    '''
    if infoStr == "RECORD":
        mR = 100
        nR = 100

        inFrame = cv2.resize(frame, (nR,mR))

        [B,G,R] = cv2.split(inFrame)
        ret3,inFrameR = cv2.threshold(R,50,255,cv2.THRESH_BINARY)

        rgbFrame = cv2.merge([inFrameR, inFrameR, inFrameR])

        test_image = image.img_to_array(rgbFrame)
        test_image = np.expand_dims(test_image, axis = 0)
        result = model.predict(test_image)

        poseIdx = np.argmax(result, axis=1)
        className = "Classified as "+str(poseIdx)





    if infoStr == "STANDBY":
        className = "..."
    '''


    mR = 100
    nR = 100

    inFrame = cv2.resize(frame, (nR,mR))

    [B,G,R] = cv2.split(inFrame)
    ret3,inFrameR = cv2.threshold(R,50,255,cv2.THRESH_BINARY)

    rgbFrame = cv2.merge([inFrameR, inFrameR, inFrameR])

    test_image = image.img_to_array(rgbFrame)
    test_image = np.expand_dims(test_image, axis = 0)
    result = model.predict(test_image)

    poseIdx = np.argmax(result, axis=1)
    print(result)
    className = "Classified as "+str(poseIdx)



    

    frameIdx = frameIdx + 1	

    cv2.putText(frame, (infoStr+" "+className), position, cv2.FONT_HERSHEY_SIMPLEX, fontSz, colour, fontTck)
    cv2.imshow("RGB",frame)
  


    #timestampStr = dateTimeObj.strftime("%H:%M:%S.%f - %b %d %Y")
    #print('Current Timestamp : ', timestampStr)

    os.system("cls")
    print(" ")
    print(" ")
    print(" ")
    print(" ")
    print(" ")
    print(" ")
    print(" ")
    print(" ")
    print(" ")
    print(" ")
    print(" ")
    print(" ")
    print(" ")
    print(" ")
    print(" ")
    print(" ")
    #print("                                %f          "%(deltaTime))
    print("                                %d     %3.4f     "%(deltaTimeInt,stdVal))
    minute = deltaTimeInt//60
    second = deltaTimeInt % 60
    hour   = deltaTimeInt//3600
    print("                               H: %d  M: %d  s: %d             "%(hour, minute, second))
    #for fileName in fileNameList:
    #	print(fileName)

    print(mAvg)
    print("                                %d               "%(sumAvg))
    print("                          %s                     "%(recordStatInfo))
    


    print(" ")
    print(" ")
    print(" ")
    print(" ")
    print(" ")
    print(" ")
    print(" ")


    #deltaTime = time.time() - start_time 
	#print(deltaTime)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()






cap.release()
cv2.destroyAllWindows()
