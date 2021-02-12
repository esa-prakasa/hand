import datetime as dtm
import os
import cv2

os.system("clear")

ctm = dtm.datetime.now()
print(ctm)

fileName = 0
secOld = 0
while(True):
    ctm = dtm.datetime.now()
    sec = ctm.second
    if (sec%2==0):
        if (sec != secOld):
            
            #os.system("clear")
            #print(" ")
            print(" ")            
            print("              %d      File %s.png has been save!  "%(sec,str(fileName)))
            fileName = fileName + 1
            
            if (fileName >4):
                fileName = 0
            
    secOld = sec
            
    