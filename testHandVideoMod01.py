import os
import numpy as np
import matplotlib.pyplot as plt

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

import cv2



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


## Video Part
videoPath = "C:\\Users\\INKOM06\\Pictures\\handwash\\mod1\\"
videoFile = os.listdir(videoPath)

videoFile[0] = "VID20200809190022.mp4"
cap = cv2.VideoCapture(videoPath+videoFile[0])


totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
print(totalFrames)

frameIdx = 0
ratio = 0.5


ret, frame0 = cap.read()
M = frame0.shape[0]
N = frame0.shape[1]

M = int(ratio*M)
N = int(ratio*N)
videoPathToSave = "C:\\Users\\INKOM06\\Pictures\\handwash\\mod1\\trdataset\\_fold1\\outvideo\\"
out = cv2.VideoWriter(videoPathToSave+"output.avi",cv2.VideoWriter_fourcc('M','J','P','G'), 30, (N,M))






while(True) and (frameIdx<(totalFrames-1)):
    ret, frame0 = cap.read()
    pct = (frameIdx/totalFrames)*100
    frame0 = cv2.rotate(frame0, cv2.ROTATE_180)

    M = frame0.shape[0]
    N = frame0.shape[1]

    mR = 100
    nR = 100

    frame = cv2.resize(frame0, (int(ratio*N),int(ratio*M)))

    #out.write(frame)

    inFrame = cv2.resize(frame0, (nR,mR))

    [B,G,R] = cv2.split(inFrame)
    ret3,inFrameR = cv2.threshold(R,50,255,cv2.THRESH_BINARY)
    cv2.namedWindow("Input frames", cv2.WINDOW_NORMAL)
    cv2.imshow("Input frames", inFrameR)


    rgbFrame = cv2.merge([inFrameR, inFrameR, inFrameR])
    

    test_image = image.img_to_array(rgbFrame)
    test_image = np.expand_dims(test_image, axis = 0)
    result = model.predict(test_image)

    

    poseIdx = np.argmax(result, axis=1)




    position = (10,40)
    frameIdxDisp = frameIdx + 10000
    frameIdxDispStr = str(frameIdxDisp)[1:]
   
    infoStr = "%2.2f"%(pct)
    infoStr =  "Frame No: "+frameIdxDispStr+" ["+infoStr+"%] Pose index: "+str(poseIdx[0])
    print(infoStr)
    
    cv2.putText(frame, infoStr, position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0, 0), 2) 
    cv2.imshow("RGB Video", frame)
    out.write(frame)



    #fileNm = str(10000 + frameIdx)
    #fileNm = fileNm[1:]+".png"

    #cv2.imwrite(path+"\\orirgb\\"+fileNm,frame)


    frameIdx +=1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break





cap.release()
cv2.destroyAllWindows()
