import numpy as np
import pandas as pd
import cv2
import random
import math
import os

os.system('cls')

csvPath = "C:\\Users\\INKOM06\\Pictures\\washhand\\csv\\"
files = os.listdir(csvPath)
csvFileNm = files[0]

print(csvFileNm)


centFileNm = "centroids.csv"

centroidFile = open((csvPath+centFileNm),"w+")
centHeadNm = ("R,G,Bl,L,A,B,S,V,")
centroidFile.write(centHeadNm+"\n")


dataOri = pd.read_csv(csvPath+csvFileNm)
print(dataOri.head(10))
print(dataOri.iloc[0:15,:])

data = dataOri.iloc[: , [0, 1, 2]].copy() 

print(data.head(10))
print(data.iloc[0:15,:])


#data = dataOri[['R','G','Bl']] 
#print(dataOri.head(10))
#print(dataOri.iloc[0:5,:])



NoS = 7
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=NoS)
kmeans.fit(data)

labels = kmeans.predict(data)
centroids = kmeans.cluster_centers_
centFloor = np.floor(centroids)

N = len(centroids)
print(str(N))


NoC = centroids.shape[0]
NoParm = centroids.shape[1]

print(NoC)
print(NoParm)

for i in range(NoC):
	cenStr = " "
	cenStrInt = " "
	cenCSV = ""
	for j in range(NoParm):
		cenStr=cenStr +"   "+"%4.2f"%(centroids[i,j])
		cenStrInt=cenStrInt +"   "+str(int(np.floor(centroids[i,j])))
		cenCSV=cenCSV +"%4.2f,"%(centroids[i,j])
	#print(cenCSV)
	print(cenStrInt)
	
	#cenCSV= cenCSV[1:]
	centroidFile.write(cenCSV+"\n")



for idx in range (NoC):
	imgc = np.zeros((200,200,3),dtype = int)
	for i in range(200):
		for j in range(200):
			imgc[i,j,0] = centroids[idx,0]
			imgc[i,j,1] = centroids[idx,1]
			imgc[i,j,2] = centroids[idx,2]

	cv2.imwrite("C:\\Users\\INKOM06\\Pictures\\washhand\\colorres\\"+str(idx)+".png", imgc)



centroidFile.close()

