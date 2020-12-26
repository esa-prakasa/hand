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


from tkinter import *
window=Tk()

import cv2




def summariseTheResult(poseCount, totalFrames):
  window.title('Summary of Handwashing')
  window.geometry("400x300+300+300")

  secondPerFrame = 1/30
  posePctAcc = 0

  poseDurationAcc = 0
  for idx in range(7):
    poseDuration = poseCount[idx]*secondPerFrame
    posePct = (poseCount[idx]/totalFrames)*100
    idxStr = "{:2d}".format(idx)


    if poseCount[idx] >= 10:
      textColour = "black"
    else:
      textColour = "red"

    text = ("Pose %s: %3.2f  sec    Pct: %3.2f %%")%(idxStr,poseDuration,posePct)
    lb0=Label(window, text=text, fg=textColour, font=("Helvetica", 12))
    yPost = 20 + idx*25
    lb0.place(x=20, y=yPost)

    poseDurationAcc = poseDurationAcc + poseDuration
    posePctAcc = posePctAcc + posePct

  posePctAcc = round(posePctAcc)
  text = ("Total Pose Duration: %3.2f  sec    Pct: %3.2f %%")%(poseDurationAcc,posePctAcc)
  lb0=Label(window, text=text, fg='black', font=("Helvetica", 12))
  yPost = 35 + yPost
  lb0.place(x=20, y=yPost)











saveVideo = False

kfold = "_fold1"

rootPath = "C:\\Users\\INKOM06\\Pictures\\handwash\\mod1\\trdataset\\"

modelPath = rootPath+kfold+"\\xmodel\\"


modelFileList = os.listdir(modelPath)
modIdx = 0
for modelFile in modelFileList:
  print(str(modIdx)+" "+modelFile)
  modIdx = modIdx + 1

modIdx = int(input("Which model that will be used? "))
#modIdx = 2


modelName = modelFileList[modIdx]
model = load_model(modelPath+modelName)

model.summary()

os.system("cls")


## Video Part
videoPath = "C:\\Users\\INKOM06\\Pictures\\handwash\\mod1\\"
videoFile = "VID20200809190022.mp4"


#videoPath = "C:\\Users\\INKOM06\\Pictures\\handwash\\hops\\"
#videoFile = "hops2.mp4"


cap = cv2.VideoCapture(videoPath+videoFile)


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
if saveVideo == True:
  out = cv2.VideoWriter(videoPathToSave+"output.avi",cv2.VideoWriter_fourcc('M','J','P','G'), 30, (N,M))




poseCount = np.zeros(7, dtype=int)

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


    poseCount[poseIdx[0]] = poseCount[poseIdx[0]] + 1

    position = (10,40)
    frameIdxDisp = frameIdx + 10000
    frameIdxDispStr = str(frameIdxDisp)[1:]
   
    infoStr = "%2.2f"%(pct)
    infoStr =  "Frame No: "+frameIdxDispStr+" ["+infoStr+"%] Pose index: "+str(poseIdx[0])
    print(infoStr)
    
    cv2.putText(frame, infoStr, position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0, 0), 2) 
    cv2.imshow("RGB Video", frame)
    
    if saveVideo == True:
      out.write(frame)



    #fileNm = str(10000 + frameIdx)
    #fileNm = fileNm[1:]+".png"

    #cv2.imwrite(path+"\\orirgb\\"+fileNm,frame)


    frameIdx +=1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break





cap.release()
cv2.destroyAllWindows()



summariseTheResult(poseCount, totalFrames)


# window.title('Summary of Handwashing')
# window.geometry("400x300+300+300")


# secondPerFrame = 1/30
# posePctAcc = 0

# poseDurationAcc = 0
# for idx in range(7):
#   poseDuration = poseCount[idx]*secondPerFrame
#   posePct = (poseCount[idx]/totalFrames)*100
#   idxStr = "{:2d}".format(idx)


#   if poseCount[idx] >= 10:
#     textColour = "black"
#   else:
#     textColour = "red"

#   text = ("Pose %s: %3.2f  sec    Pct: %3.2f %%")%(idxStr,poseDuration,posePct)
#   lb0=Label(window, text=text, fg=textColour, font=("Helvetica", 12))
#   yPost = 20 + idx*25
#   lb0.place(x=20, y=yPost)

#   poseDurationAcc = poseDurationAcc + poseDuration
#   posePctAcc = posePctAcc + posePct


# posePctAcc = round(posePctAcc)
# text = ("Total Pose Duration: %3.2f  sec    Pct: %3.2f %%")%(poseDurationAcc,posePctAcc)
# lb0=Label(window, text=text, fg='black', font=("Helvetica", 12))
# yPost = 20 + idx*25
# lb0.place(x=20, y=yPost)


#textVal = str(frameIdx)

#lbl=Label(window, text=textVal, fg='red', font=("Helvetica", 16))
#lbl.place(x=60, y=50)

window.mainloop()
cv2.destroyAllWindows()

