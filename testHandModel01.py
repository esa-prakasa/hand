import os
import numpy as np
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.callbacks import ModelCheckpoint

from keras.models import load_model
from keras.preprocessing import image
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels



kfold = "_fold1"

#rootPath = "C:\\Users\\INKOM06\\Pictures\\washhand\\trainData\\huebEr\\ds\\"
rootPath = "C:\\Users\\INKOM06\\Pictures\\handwash\\mod1\\trdataset\\"

#rootPath  = "C:\\Users\\INKOM06\\Pictures\\roadDataset\\bandung\\dataset\\"+kfold+"\\train\\"
modelPath = rootPath+kfold+"\\xmodel\\"


modelFileList = os.listdir(modelPath)
modIdx = 0
for modelFile in modelFileList:
  print(str(modIdx)+" "+modelFile)
  modIdx = modIdx + 1

modIdx = int(input("Which model that will be used? "))


#modelName = "hand_best_model_"+kfold+".h5"
modelName = modelFileList[modIdx]
model = load_model(modelPath+modelName)

model.summary()

testFolder =[]
testFolder.append(rootPath+kfold+"\\test\\0\\")
testFolder.append(rootPath+kfold+"\\test\\1\\")
testFolder.append(rootPath+kfold+"\\test\\2\\")
testFolder.append(rootPath+kfold+"\\test\\3\\")
testFolder.append(rootPath+kfold+"\\test\\4\\")
testFolder.append(rootPath+kfold+"\\test\\5\\")
testFolder.append(rootPath+kfold+"\\test\\6\\")

dimSz = 100
#os.system("cls")
accuracy = []

y_test = []
y_pred = []

for clsSelected in range (7):
	testedFolder = testFolder[clsSelected]

	files = os.listdir(testedFolder)
	NF = len(files)

#	print(str(clsSelected)+" "+str(NF))
	print(str(clsSelected))

	target_names = [item for item in os.listdir(rootPath) if os.path.isdir(os.path.join(rootPath, item))]

	correctCount = 0

	for i in range(NF): #range(Nb):
		test_image = image.load_img((testedFolder+files[i]), target_size =(dimSz, dimSz,3))
		test_image = image.img_to_array(test_image)
		
		
		test_image = np.expand_dims(test_image, axis = 0)
		result = model.predict(test_image)

		print(result)
		result2 = result[0]
		#print((result2))
		classIdx = np.argmax(result2)
		print(str(i)+" --- "+files[i]+"--"+str(clsSelected)+" >>> "+str(classIdx)+" --> "+str(result2))
		y_test.append(str(clsSelected))
		y_pred.append(str(classIdx))
		if (clsSelected==classIdx):
			correctCount = correctCount + 1

	accuracy.append(correctCount/NF)

print("  ")
for clsSelected in range(7):	
	print("Class of "+str(clsSelected)+(" %3.2f"%(accuracy[clsSelected])))


#cm = []
#cm.append([accuracy[0], (1-accuracy[0])])
#cm.append([(1-accuracy[1]), accuracy[1]])


cm = confusion_matrix(y_test, y_pred)
print(cm)




def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes =  ['0', '1', '2', '3', '4', '5']# classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    plt.rc('font', size=15)
    plt.rc('axes', titlesize=12)
    plt.rc('axes', labelsize=12)
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    plt.rc('legend', fontsize=20)   

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax







class_names = ['0', '1', '2', '3', '4', '5', '6']
plot_confusion_matrix(y_test, y_pred, classes=class_names,
                      title='Confusion matrix, without normalization')
plt.show()




### =============================
#Another statistical parameters:
### =============================

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt




acc   = accuracy_score(y_test,y_pred)
prec  = precision_score(y_test,y_pred, average =None)
recl  = recall_score(y_test,y_pred, average =None)
f1_sc = f1_score(y_test,y_pred, average =None)
cohkp = cohen_kappa_score(y_test,y_pred)
#roc   = roc_auc_score(y_test,y_pred)

print("Accuracy          : %3.4f "%(acc))
print("Average precision : %3.4f "%(np.average(prec)))
print("Aveage recall    : %3.4f "%(np.average(recl)))
print("Average F1-score  : %3.4f "%(np.average(f1_sc)))
print("Cohen-Kappa score : %3.4f "%(cohkp))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print(cm)

#className = []
#for i in range(10):
#    className.append(str(i))

className = class_names# ["Border", "Non Border"]


cmap=plt.cm.Reds
fig, ax = plt.subplots()
im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
ax.figure.colorbar(im, ax=ax)

ax.figure.set_size_inches(8,6,True)

ax.set(xticks=np.arange(cm.shape[1]),
  yticks=np.arange(cm.shape[0]),
  xticklabels=className, yticklabels=className,
  title='',
  ylabel='Input Area',
  xlabel='Predicted Area')

# Rotate the tick labels and set their alignment.
#plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
#  rotation_mode="anchor")

fmt = 'd'
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
  for j in range(cm.shape[1]):
    ax.text(j, i, format(cm[i, j], fmt),
    ha="center", va="center",
    color="white" if cm[i, j] > thresh else "black")

#fig.tight_layout()

plt.show()
