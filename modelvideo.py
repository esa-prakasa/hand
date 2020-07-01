import numpy as np
import cv2
import os


os.system("cls")

modelPath = "C:\\Users\\INKOM06\\Pictures\\washhand\\xmodel\\"
csvPath = "C:\\Users\\INKOM06\\Pictures\\washhand\\csv\\"

videoFile = os.listdir(modelPath)
#videoFile = "oreclip.mp4"


csvFileNm = "dataset.csv"
csvFile = open((csvPath+csvFileNm),"w+")

outVal = ("No, i, j,  r,  g, b, class")
csvFile.write(outVal+"\n")




cap = cv2.VideoCapture(modelPath+videoFile[0])

totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
print(totalFrames)

frameIdx = 0
ratio = 0.6





def updateCSV(idxCsv, imgRGB, imgLAB, imgHSV, csvObj):
    M = imgRGB.shape[0] 
    N = imgRGB.shape[1]
    for i in range(M):
        for j in range(N):
            R = imgRGB[i,j,2]
            G = imgRGB[i,j,1]
            Bl = imgRGB[i,j,0]
            outVal = str(R)+","+str(G)+","+str(Bl)+", "

            L = imgLAB[i,j,2]
            A = imgLAB[i,j,1]
            B = imgLAB[i,j,0]
            outVal = outVal+ str(L)+","+str(A)+","+str(B)+", "

            H = imgHSV[i,j,2]
            S = imgHSV[i,j,1]
            V = imgHSV[i,j,0]
            outVal = outVal+ str(H)+","+str(S)+","+str(V)

            idxCsv = idxCsv + 1 
            outVal = str(idxCsv)+": "+outVal 

            print(outVal)

            #outVal = ("No, i, j,  r,  g, b, class")
            #csvRoadFile.write(outVal+"\n")
    return idxCsv





#totalFrames = 100

#csvRoadFile = open((pathToSave+"csvfiles\\"+csvRoadFileNm),"w+")

#outVal = ("No, i, j,  r,  g, b, class")
#csvRoadFile.write(outVal+"\n")

totalFarmes = 10

idxCsv = 0



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

    mm2 = mm // 2
    nn2 = nn // 2

    #print("mm2 "+str(mm2))
    #print("nn2 "+str(nn2))

    #cv2.imshow("Ori",frame)



    delta = min(mm2,nn2)
    delta = int(0.4*delta)
    frame = frame[mm2-delta:mm2+delta,nn2-delta:nn2+delta,:]


    labImg = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    hsvImg = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    idxCsv = updateCSV(idxCsv, frame, labImg, hsvImg, csvFile)



    [Blue,G,R] = cv2.split(frame)
    [L,A,B] = cv2.split(labImg)
    [H,S,V] = cv2.split(hsvImg)

    #img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    frameIdx = frameIdx + 1

    rgbMerge = np.hstack((R,G,Blue)) 
    labMerge = np.hstack((L,A,B)) 
    hsvMerge = np.hstack((H,S,V)) 

    #oriMerge = np.hstack((frame,frame,frame))
    #rgbMerge = np.hstack((R,G,Blue))
    #labMerge = np.hstack((L,A,B))
    #hsvMerge = np.hstack((H,S,V))

    finMerge = np.vstack((rgbMerge,labMerge, hsvMerge))

    #H = cv2.blur(H,(5,5))

    #cv2.imshow("All channels", finMerge)




    #Hb = cv2.inRange(H, 0, 15)
    #m3 = Hb.shape[0]
    #n3 = Hb.shape[1]
    #rgbMs = np.zeros([m3,n3,3],dtype="uint8")

    #for ii in range(m3):
    #    for jj in range(n3):
    #        if (Hb[ii,jj] == 255):
    #            rgbMs[ii,jj,:] = frame[ii,jj,:]

    #cv2.imshow("Binary HUE", Hb)
    #cv2.imshow("RGB", frame)
    #cv2.imshow("Masked of RGB", rgbMs)

    if ((frameIdx%10)==0):
        fileNm = str(10000 + frameIdx)
        fileNm = fileNm[1:]+".png"
        print(fileNm)

        #cv2.imwrite(path+"\\hue\\"+fileNm,H)
        #cv2.imwrite(path+"\\hueb\\"+fileNm,Hb)
        #cv2.imwrite(path+"\\rgb\\"+fileNm,frame)
        #cv2.imwrite(path+"\\mask\\"+fileNm,rgbMs)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


csvFile.close()

cap.release()
cv2.destroyAllWindows()

'''
    b1 = int(0.65*mm)
    b2 = int(0.95*mm)
    B_Near = Blue[b1:b2,:]
    S_Near = S[b1:b2,:]
    finMerge2 = np.vstack((B_Near, S_Near))
    
    b1 = int(0.6*mm)
    b2 = int(0.7*mm)
    B_Far = Blue[b1:b2,:]
    S_Far = S[b1:b2,:]
    finMerge2b = np.vstack((B_Far, S_Far))


    #cv2.imshow('Original images',oriMerge[int(0.5*mm):int(0.7*mm),:,:])
    cv2.imshow('RGB and LAB images',finMerge)
    cv2.imshow('Near sides',finMerge2)
    cv2.imshow('Far sides',finMerge2b)


    ret, Bbw = cv2.threshold(B_Near,150,255,cv2.THRESH_BINARY)
    ret, Sbw = cv2.threshold(S_Near,230,255,cv2.THRESH_BINARY)

    Smask = cv2.inRange(S_Near, 80, 150)


    finMerge3 = np.vstack((Bbw, Sbw, Smask))
    cv2.imshow('Binary Both sides',finMerge3)

'''

