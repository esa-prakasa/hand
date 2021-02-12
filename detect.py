import cv2
import os
import numpy as np


os.system("clear")

cap0 = cv2.VideoCapture(0)

print("done")
frameNo = 0
textPost = (10,50)

ret, frame0 = cap0.read()
M = frame0.shape[0]
N = frame0.shape[1]
ratio = 0.2
m1 = int(M*ratio)
n1 = int(N*ratio)


refImg0 = np.zeros((m1,n1), np.uint8)
delta = np.zeros((m1,n1), np.uint8)

if (cap0.isOpened()):
    while(True):

        ret, frame0 = cap0.read()
        #frame0 = cv2.resize(frame0, (n1, m1), interpolation = cv2.INTER_AREA)
        refImg = cv2.resize(frame0, (n1, m1), interpolation = cv2.INTER_AREA)
        refImg = cv2.cvtColor(refImg, cv2.COLOR_BGR2GRAY)
        
        
        for i in range (m1):
            for j in range (n1):
                #print("%d  %d   %d"%(i,j,refImg0[i,j]))

                #print("%d  %d   %f"%(i,j,float(refImg[i,j])))
                #print("%d  %d   %f"%(i,j,float(refImg0[i,j])))

                deltaVal = int(float(refImg[i,j]) - float(refImg0[i,j]))
                if deltaVal<0:
                    deltaVal = 0                
                    #cv2.putText(frame0, "OFF", textPost,cv2.FONT_HERSHEY_SIMPLEX,
                    #1, (0, 0, 255, 255), 1)
                    
                if deltaVal>10:
                    deltaVal = 255
                    cv2.putText(frame0, "ON", textPost,cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0, 255), 1)
        
                delta[i,j] = deltaVal
                    

        
        refImg0 = refImg
                
        

        cv2.imshow("Original image", frame0)
        cv2.imshow("Ref image", refImg)
        cv2.imshow("Delta image", delta)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            cap0.release()
            cv2.destroyAllWindows()
            break


