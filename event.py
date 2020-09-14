import numpy as np
import cv2
import os
import time
import pandas as pd 
from datetime import datetime
import math
import random

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


cap = cv2.VideoCapture(0)
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
    	fontSz = 1.5
    	fontTck = 4
    else:
    	infoStr = "STANDBY"
    	colour = [0, 255, 0, 0]
    	fontSz = 1.5
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


    

    frameIdx = frameIdx + 1	

    cv2.putText(frame, infoStr, position, cv2.FONT_HERSHEY_SIMPLEX, fontSz, colour, fontTck)
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
