import numpy as np
import cv2
import os
import time

start_time = time.time()



os.system("cls")

modelPath = "C:\\Users\\INKOM06\\Pictures\\washhand\\xmodel\\"
csvPath = "C:\\Users\\INKOM06\\Pictures\\washhand\\csv\\"

videoFile = os.listdir(modelPath)

csvFileNm = "dataset.csv"
csvFile = open((csvPath+csvFileNm),"w+")

outVal = ("R, G, Bl,  L,  A, B, S, V")
csvFile.write(outVal+"\n")



cap = cv2.VideoCapture(modelPath+videoFile[0])

totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
print(totalFrames)

frameIdx = 0
ratio = 0.3
sampPix = 30


def updateCSV(frameIdx, pct, idxCsv, imgRGB, imgLAB, imgHSV, csvFile):
    M = imgRGB.shape[0] 
    N = imgRGB.shape[1]
    for i in range(M):
        for j in range(N):

            if (((i%sampPix)==0) and ((j%sampPix)==0)):
                R = imgRGB[i,j,2]
                G = imgRGB[i,j,1]
                Bl = imgRGB[i,j,0]
                outVal = str(R)+","+str(G)+","+str(Bl)+", "

                L = imgLAB[i,j,2]
                A = imgLAB[i,j,1]
                B = imgLAB[i,j,0]
                outVal = outVal+ str(L)+","+str(A)+","+str(B)+", "

                #H = imgHSV[i,j,2]
                S = imgHSV[i,j,1]
                V = imgHSV[i,j,0]
                outVal = outVal+ str(S)+","+str(V)

                outValforCSV = outVal

                idxCsv = idxCsv + 1 
                outVal = str(idxCsv)+": "+outVal 

                pctStr = ("%4.2f"%(pct))

                outDisp = pctStr+" % ---> "+outVal
                outDisp = str(frameIdx)+" ==> "+outDisp
                print(outDisp)

                csvFile.write(outValforCSV+"\n")
    return idxCsv



#totalFarmes = 2

idxCsv = 0



while(True) and (frameIdx<(totalFrames-1)):
    ret, frame = cap.read()
    pct = (frameIdx/totalFrames)*100

    M = frame.shape[0]
    N = frame.shape[1]

    frame = cv2.resize(frame, (int(ratio*N),int(ratio*M)))

    mm = frame.shape[0]
    nn = frame.shape[1]

    mm2 = mm // 2
    nn2 = nn // 2


    delta = min(mm2,nn2)
    delta = int(0.4*delta)
    frame = frame[mm2-delta:mm2+delta,nn2-delta:nn2+delta,:]


    labImg = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    hsvImg = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    idxCsv = updateCSV(frameIdx, pct, idxCsv, frame, labImg, hsvImg, csvFile)

    cv2.imshow("RGB",frame)

    #[Blue,G,R] = cv2.split(frame)
    #[L,A,B] = cv2.split(labImg)
    #[H,S,V] = cv2.split(hsvImg)

    #img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    frameIdx = frameIdx + 1

    #rgbMerge = np.hstack((R,G,Blue)) 
    #labMerge = np.hstack((L,A,B)) 
    #hsvMerge = np.hstack((H,S,V)) 


    #finMerge = np.vstack((rgbMerge,labMerge, hsvMerge))


    #if ((frameIdx%10)==0):
    #    fileNm = str(10000 + frameIdx)
    #    fileNm = fileNm[1:]+".png"
    #    print(fileNm)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


csvFile.close()


deltaTime = time.time() - start_time 
print("--- %5.5s seconds ---" % (deltaTime))




cap.release()
cv2.destroyAllWindows()

