import numpy as np
import cv2
import os


os.system("cls")
path = "C:\\Users\\INKOM06\\Pictures\\washhand\\hueb\\"
fileList = os.listdir(path)

#print(path+files[1])

#imgIdx = 3

fileNm = "0730"
#img = cv2.imread(path+fileList[imgIdx])
#img = cv2.imread(path+"0400.jpg")

img = cv2.imread(path+fileNm+".jpg")


img0 = img.copy()
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#img = cv2.imread('/home/img/python.png', cv2.IMREAD_UNCHANGED)

M = img.shape[0]
N = img.shape[1]

print(M)
print(N)
print(img.shape)
#ratio = 0.5

print(img[47,80])
#img = cv2.resize(img,(int(ratio*M), int(ratio*N)) , interpolation = cv2.INTER_AREA)   




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
	iC = int(iC/Nimg)
	jC = int(jC/Nimg)
	return iC,jC



[iC, jC] = getCentroidOfMass(img)


#img = cv2.imread(path+files[1])
cv2.imshow("img", img )


for i in range(-5,6,1):
	for j in range(-5,6,1):
		img0[iC+i, jC+j, 0] = 0
		img0[iC+i, jC+j, 1] = 0
		img0[iC+i, jC+j, 2] = 255



cv2.imshow("img", img0 )






cv2.waitKey(0)
cv2.destroyAllWindows()
