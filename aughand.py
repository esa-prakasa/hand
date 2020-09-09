import os
import cv2
# example of horizontal shift image augmentation
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np

os.system("cls")

clsIdx = input("What is the class index? ")

srcFolder = "C:\\Users\\INKOM06\\Pictures\\handwash\\mod1\\dataset\\"+clsIdx+"\\"
tgtFolder = "C:\\Users\\INKOM06\\Pictures\\handwash\\mod1\\dataset\\_"+clsIdx+"\\"

srcFiles = os.listdir(srcFolder)
tgtFiles = []

NsF = len(srcFiles)

NsFperm = np.random.permutation(NsF)


NinFrame = int(input("How many original frames required? "))

#for i in range(len(srcFiles)):
for i in range(NinFrame):
	#print(srcFiles[i])

	frameIdx = NsFperm[i]
#	fNm = srcFiles[i]
	fNm = srcFiles[frameIdx]
	fNm = fNm[:-4]
	tgtFiles.append(fNm)


#for idx in range(len(srcFiles)):
for idx in range(NinFrame):
	print(tgtFiles[idx])
	frameIdx = NsFperm[idx]
#	img = load_img(srcFolder+srcFiles[idx])
	img = load_img(srcFolder+srcFiles[frameIdx])
	data = img_to_array(img)
	samples = expand_dims(data, 0)
	shftPix = 3

	datagen = ImageDataGenerator(
		height_shift_range=[-shftPix,shftPix],
		width_shift_range=[-shftPix,shftPix],
		rotation_range=10,
		#brightness_range=[0.5,1.0],
		zoom_range=[0.5,1.0])
	it = datagen.flow(samples, batch_size=1)

	for i in range(10):
		batch = it.next()
		image = batch[0].astype('uint8')
		image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

		idxS = str(100+i)
		idxS = idxS[1:]

		fileName = "ag_"+tgtFiles[idx]+"_"+idxS+".png"
		pathToSave = tgtFolder+fileName
		print(pathToSave)
		cv2.imwrite(pathToSave, image)

