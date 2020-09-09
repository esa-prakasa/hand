import numpy as np
import cv2
import os


os.system("cls")

sourcePath = "C:\\Users\\INKOM06\\Pictures\\handwash\\mod1\\trdataset\\"


minL = 1e3
for i in range(6):
	path = sourcePath+str(i)+"\\"
	files = os.listdir(path)
	N = len(files)
	print(N)
	if N<minL:
		minL = N




rTr = 0.7
rVl = 0.2

nTr = round(0.7*minL)
nVl = round(0.2*minL)
nTs = minL - nTr - nVl

#idx = []
#for i in range(minL):
#	idx.append(i)

#readPath = "C:\\Users\\INKOM06\\Pictures\\washhand\\trainData\\huebEr\\" 
#trPath = "C:\\Users\\INKOM06\\Pictures\\washhand\\trainData\\huebEr\\train\\" 
#vlPath = "C:\\Users\\INKOM06\\Pictures\\washhand\\trainData\\huebEr\\valid\\" 
#tsPath = "C:\\Users\\INKOM06\\Pictures\\washhand\\trainData\\huebEr\\test\\" 

readPath = "C:\\Users\\INKOM06\\Pictures\\handwash\\mod1\\trdataset\\" 
trPath = "C:\\Users\\INKOM06\\Pictures\\handwash\\mod1\\trdataset\\fold1\\train\\" 
vlPath = "C:\\Users\\INKOM06\\Pictures\\handwash\\mod1\\trdataset\\fold1\\valid\\" 
tsPath = "C:\\Users\\INKOM06\\Pictures\\handwash\\mod1\\trdataset\\fold1\\test\\" 

foldNo = "fold1"






for idxCls in range(7):
	idx = np.random.permutation(minL)
	trIdx = (idx[0:nTr])
	vlIdx =idx[nTr:(nTr+nVl)]
	tsIdx = idx[-nTs:]
	
	filesNm = os.listdir(readPath+str(idxCls)+"\\")
	
	for j in range(len(trIdx)):
		pathToRead = (readPath+str(idxCls)+"\\"+filesNm[trIdx[j]])
		img = cv2.imread(pathToRead)
		pathToSave = (readPath+foldNo+"\\train\\"+str(idxCls)+"\\"+filesNm[trIdx[j]])
		cv2.imwrite(pathToSave,img)
		#print(pathToRead)
		#print(pathToSave)


	for j in range(len(vlIdx)):
		pathToRead = (readPath+str(idxCls)+"\\"+filesNm[vlIdx[j]])
		img = cv2.imread(pathToRead)
		pathToSave = (readPath+foldNo+"\\valid\\"+str(idxCls)+"\\"+filesNm[vlIdx[j]])
		cv2.imwrite(pathToSave,img)
		#print(pathToSave)

	for j in range(len(tsIdx)):
		pathToRead = (readPath+str(idxCls)+"\\"+filesNm[tsIdx[j]])
		img = cv2.imread(pathToRead)
		pathToSave = (readPath+foldNo+"\\test\\"+str(idxCls)+"\\"+filesNm[tsIdx[j]])
		cv2.imwrite(pathToSave,img)
		#print(pathToSave)







print(" ")
print(nTr)
print(nVl)
print(nTs)
print(minL)












cv2.waitKey(0)
cv2.destroyAllWindows()
