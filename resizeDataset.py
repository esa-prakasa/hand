import os
import cv2


os.system("cls")


#inputPath = "C:\\Users\\INKOM06\\Pictures\\handwash\\mod1\\RTh\\"
inputPath = "C:\\Users\\INKOM06\\Pictures\\handwash\\mod1\\dataset\\"



pose = 0


for idx in range(1,7):
	pose = idx
	fileList = os.listdir(inputPath+str(pose)+"\\")
	for files in fileList:
		img = cv2.imread(inputPath+str(pose)+"\\"+files)
		#M = img.shape[0]
		#N = img.shape[1]
		#print(M,N)
		#R = min(M,N)
		#print(R)
		R = 120
		#print(R)
		img = cv2.resize(img, (R,R), interpolation = cv2.INTER_AREA)
		cv2.imwrite(inputPath+str(pose)+"\\"+files, img)
		#M = img.shape[0]
		#N = img.shape[1]
		#print(M,N)
		print("%d %s has been resized" %(idx,files))
		idx = idx + 1
		#cv2.imshow("image resized",imgR)


cv2.waitKey(0)
cv2.destroyAllWindows()

