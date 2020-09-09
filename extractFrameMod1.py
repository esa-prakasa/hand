# Dataset Model1 video
# Esa's hand with green background
# The data was taken on 9 August 2020

import numpy as np
import cv2
import os
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
from skimage import morphology, filters



os.system("cls")

path = "C:\\Users\\INKOM06\\Pictures\\handwash\\mod1\\"
videoFile = os.listdir(path)
#videoFile = "oreclip.mp4"

videoFile[0] = "VID20200809190022.mp4"
cap = cv2.VideoCapture(path+videoFile[0])



totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
print(totalFrames)

frameIdx = 0


def getCentroidOfMass(img):
    M = img.shape[0]
    N = img.shape[1]
    iC = 0
    jC = 0
    Nimg = 0
    for i in range(M):
        for j in range(N):
            if (img[i,j] == 255):
                Nimg = Nimg + 1
                iC = iC + i
                jC = jC + j
    
    if Nimg>0:
        iC = int(iC/Nimg)
        jC = int(jC/Nimg)
    if Nimg == 0:
        iC = M//2
        jC = N//2

    return iC,jC



def createHbwithCent(img, iC, jC):
    M = img.shape[0]
    N = img.shape[1]
    HbC = np.zeros((M,N,3), dtype=int)

    for i in range(M):
        for j in range(N):
            for k in range(3):
                HbC[i,j,k] = img[i,j]

    for i in range(-3,4,1):
        for j in range(-3,4,1):
            HbC[iC+i, jC+j, 0] = 0
            HbC[iC+i, jC+j, 1] = 0
            HbC[iC+i, jC+j, 2] = 255


    return HbC


def allSkel(skel):
    M = skel.shape[0]
    N = skel.shape[1]
    finVal = False
    for i in range(M):
        for j in range(N):
            finVal = finVal or skel[i,j]
    return finVal


ratio = 0.2


while(True) and (frameIdx<(totalFrames-1)):
    ret, frame = cap.read()
    pct = (frameIdx/totalFrames)*100

    M = frame.shape[0]
    N = frame.shape[1]

    frame = cv2.resize(frame, (int(ratio*N),int(ratio*M)))

    print(pct)

    frame = cv2.rotate(frame, cv2.ROTATE_180)


    cv2.imshow("RGB Video", frame)


    fileNm = str(10000 + frameIdx)
    fileNm = fileNm[1:]+".png"

    cv2.imwrite(path+"\\orirgb\\"+fileNm,frame)


    frameIdx +=1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


'''
while(True) and (frameIdx<(totalFrames-1)):
    ret, frame = cap.read()
    pct = (frameIdx/totalFrames)*100

    M = frame.shape[0]
    N = frame.shape[1]

    frame = cv2.resize(frame, (int(ratio*N),int(ratio*M)))


    mm = frame.shape[0]
    nn = frame.shape[1]

    #print(mm)
    #print(nn)

    mPt = round(mm*0.5)#mm // 2
    nPt = round(nn*0.45)#nn // 2

    #print(mm2)
    #print(nn2)


    delta = min(mm,nn)

    #print(delta)
    #delta = int(0.5*delta)
    delta = int(0.5*delta)
    print(str(frameIdx)+" of "+str(totalFrames)) 
#    frame = frame[mm2-delta:mm2+delta,nn2-delta:nn2+delta,:]
    frame = frame[mPt-delta:mPt+delta,nPt-delta:nPt+delta,:]


    labImg = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    hsvImg = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)



    [Blue,G,R] = cv2.split(frame)
    [L,A,B] = cv2.split(labImg)
    [H,S,V] = cv2.split(hsvImg)

    #img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    frameIdx = frameIdx + 1


    H = cv2.blur(H,(3,3))
    S = cv2.blur(S,(3,3))



    oriMerge = np.hstack((frame,frame,frame))
    rgbMerge = np.hstack((R,G,Blue))
    labMerge = np.hstack((L,A,B))
    hsvMerge = np.hstack((H,S,V))

    finMerge = np.vstack((rgbMerge,labMerge, hsvMerge))



    #cv2.imshow("HUE", H)
    cv2.imshow("Fin Merge", finMerge)



    ret, St = cv2.threshold(S, 10, 255,cv2.THRESH_BINARY)
    cv2.imshow("S threshold", St)



    #Hb = cv2.inRange(H, 0, 15)
    m3 = St.shape[0]
    n3 = St.shape[1]
    rgbMs = np.zeros([m3,n3,3],dtype="uint8")

    for ii in range(m3):
        for jj in range(n3):
            if (St[ii,jj] == 255):
                rgbMs[ii,jj,:] = frame[ii,jj,:]











    if (frameIdx!=0):
        fileNm = str(10000 + frameIdx)
        fileNm = fileNm[1:]+".png"
        print(fileNm)


        cv2.imwrite(path+"\\hue\\"+fileNm,H)
        cv2.imwrite(path+"\\S\\"+fileNm,S)
        cv2.imwrite(path+"\\St\\"+fileNm,St)
        cv2.imwrite(path+"\\rgb\\"+fileNm,frame)
        cv2.imwrite(path+"\\mask\\"+fileNm,rgbMs)
#        cv2.imwrite(path+"\\cent\\"+fileNm,HbC)







    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

'''

'''
    Hb = cv2.inRange(H, 0, 15)
    m3 = Hb.shape[0]
    n3 = Hb.shape[1]
    rgbMs = np.zeros([m3,n3,3],dtype="uint8")

    for ii in range(m3):
        for jj in range(n3):
            if (Hb[ii,jj] == 255):
                rgbMs[ii,jj,:] = frame[ii,jj,:]

    



   
    [iC0, jC0] = getCentroidOfMass(Hb)

    HbC = createHbwithCent(Hb,iC0, jC0)


    kernel = np.ones((2, 2), np.uint8) 
    HbErd = cv2.erode(Hb, kernel)  

            

    cv2.imshow("Binary HUE", Hb)
    cv2.imshow("Hb eroded", HbErd)
    cv2.imshow("RGB", frame)

'''
    



cap.release()
cv2.destroyAllWindows()

