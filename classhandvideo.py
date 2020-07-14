# classification for hand pose movement

import numpy as np
import cv2
import os
import time
import pandas as pd 

start_time = time.time()
saveFrames = True
saveOutVideo = True


os.system("cls")

modelPath = "C:\\Users\\INKOM06\\Pictures\\washhand\\xmodel\\"
#modelPath = "C:\\Users\\INKOM06\\Pictures\\washhand\\xmodel\\_rgb\\"
csvPath = "C:\\Users\\INKOM06\\Pictures\\washhand\\csv\\"
framesPath = "C:\\Users\\INKOM06\\Pictures\\washhand\\xmodel\\frames\\"
videoPath = "C:\\Users\\INKOM06\\Pictures\\washhand\\xmodel\\video\\"

videoFile = os.listdir(modelPath)
print(videoFile)

centFileNm = "centroids.csv"


cent = pd.read_csv(csvPath+centFileNm) 

#print(cent.head(7))

#print(cent.iloc[0,0])


#print("-----")

#print(cent.iloc[0,0])

videoIdx = 2
cap = cv2.VideoCapture(modelPath+videoFile[videoIdx])

frameName = videoFile[videoIdx]
frameName = frameName[:-4]



totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
#print(totalFrames)

frameIdx = 0
ratio = 0.5
sampPix = 30


def classify(imgRGB,imgLAB,imgHSV, cent):
    M = imgRGB.shape[0]
    N = imgLAB.shape[1]
    classAcc   = np.zeros([7],dtype=int)
#    classDelta = np.zeros([7],dtype=int)
    inpPat = []
    for i in range(0,M,sampPix):
        for j in range(0,N,sampPix):
            classDelta = np.zeros([7],dtype=int)
            
            inpPat.append(imgRGB[i,j,2])
            inpPat.append(imgRGB[i,j,1])
            inpPat.append(imgRGB[i,j,0]) 

            inpPat.append(imgLAB[i,j,0]) 
            inpPat.append(imgLAB[i,j,1]) 
            inpPat.append(imgLAB[i,j,2]) 

            inpPat.append(imgHSV[i,j,1]) 
            inpPat.append(imgHSV[i,j,2]) 

            nPat = 3
            for k in range(len(classAcc)):
                for m in range(nPat):
                    classDelta[k] = classDelta[k] +  (inpPat[m] - cent.iloc[k,m])**2
                classDelta[k] = np.sqrt(classDelta[k])/nPat

            minDelta = min(classDelta)
            
            for k in range(len(classAcc)):
                if (classDelta[k] == minDelta):
                    classIdx = k 
            classAcc[classIdx] = classAcc[classIdx] + 1

            #print(str(k))

    maxClassAcc = max(classAcc)
    for k in range(len(classAcc)):
        if classAcc[k] == maxClassAcc:
            finalClass = k




    return finalClass





colourNameList = ["Yellow","Black","Green","Blue","Violet","No action","Red"] 

idxCsv = 0
classFrame = np.zeros((7),dtype = int)
#totalFrames = 10

classSet =[]

if (saveOutVideo==True):
    ret, frame = cap.read()
    M = int(ratio*frame.shape[0])
    N = int(ratio*frame.shape[1])

    #M = int(ratio*M)
    #N = int(ratio*N)
    out = cv2.VideoWriter((videoPath+"output.avi"),cv2.VideoWriter_fourcc('M','J','P','G'), 30, (N,M))

while(True) and (frameIdx<(totalFrames-1)):
    ret, frame = cap.read()
    pct = (frameIdx/totalFrames)*100


    M = frame.shape[0]
    N = frame.shape[1]

    frame = cv2.resize(frame, (int(ratio*N),int(ratio*M)))
    frame0 = frame.copy()

    mm = frame.shape[0]
    nn = frame.shape[1]

    mm2 = mm // 2
    nn2 = nn // 2


    delta = min(mm2,nn2)
    delta = int(0.4*delta)
    frame = frame[mm2-delta:mm2+delta,nn2-delta:nn2+delta,:]


    labImg = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    hsvImg = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #idxCsv = updateCSV(frameIdx, pct, idxCsv, frame, labImg, hsvImg, csvFile)
    finalClass = classify(frame,labImg,hsvImg, cent)

    if (finalClass in classSet) == False:
        classSet.append(finalClass) 

    classFrame[finalClass] = classFrame[finalClass] + 1



    colourNm = colourNameList[finalClass]



    [Blue,G,R] = cv2.split(frame)
    [L,A,B]    = cv2.split(labImg)
    [H,S,V]    = cv2.split(hsvImg)


    position = (10,40)
    
    start_point = (5,10)
    end_point   = (280,50)

    color = (255, 255, 255) 
   
    thickness = -1
   
    frame0 = cv2.rectangle(frame0, start_point, end_point, color, thickness) 
    cv2.putText(frame0, (str(frameIdx)+" "+str(finalClass)+" "+colourNm), position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0, 0), 2) 


    cv2.imshow("Original", frame0)
    cv2.imshow("RGB", frame)

    if (saveFrames == True):
        frameIdxStr = str(100000+frameIdx)
        frameIdxStr = frameIdxStr[1:]
   
        finFrameName = frameName+"__"+frameIdxStr+".png"
        cv2.imwrite(framesPath+finFrameName, frame0)

    if (saveOutVideo == True):
        out.write(frame0)
    



    #img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    frameIdx = frameIdx + 1



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break




deltaTime = time.time() - start_time 
timePerFrame = deltaTime/totalFrames
#print("--- %5.5s seconds ---" % (deltaTime))
#print("Time per frame: %5.5s seconds ---" % (timePerFrame))

print(" Pose      Frames     Time ")
for i in range(len(classFrame)):
    cN = " {:<10}".format(colourNameList[i])
    classFrameStr = "{:<8}".format(str(classFrame[i]))
    print("%s  %s  %4.2f s"%(cN,classFrameStr,(classFrame[i]*timePerFrame)))

print(" ")
print(" The completed steps: ")
for i in range(len(classSet)):
    print(" "+str(i)+" "+colourNameList[classSet[i]])

uncondSteps = []
for i in range(len(classSet)):
    if (i in classSet)==False:
        uncondSteps.append(i)
print(" ")
print(" The unconducted steps: ")

if (len(uncondSteps)>0):
    for i in range(len(uncondSteps)):
        print(" "+colourNameList[uncondSteps[i]])
if (len(uncondSteps)==0):
    print(" "+"None")



cap.release()
cv2.destroyAllWindows()
