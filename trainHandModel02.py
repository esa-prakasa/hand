import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.callbacks import ModelCheckpoint
import cv2

import tensorflow as tf
from tensorflow import keras

os.system("cls")




#img_width, img_height = 20, 20
#img_width, img_height = 108, 108
img_width, img_height = 200, 200

kfold = "_fold1"

rootPath = "C:\\Users\\INKOM06\\Pictures\\handwash\\mod1\\trdataset\\"

trainFolder = rootPath+kfold+"\\train\\"
validFolder = rootPath+kfold+"\\valid\\"




train_data_dir      = trainFolder
validation_data_dir = validFolder


filesInFolder = os.listdir(trainFolder)
nb_train_samples = len(filesInFolder)
print(nb_train_samples)

filesInFolder = os.listdir(validFolder)
nb_validation_samples = len(filesInFolder)
print(nb_validation_samples)

epochs = 50
batch_size = 15


#if K.image_data_format() == 'channels_first':
#    input_shape = (3, img_width, img_height)
#else:
#    input_shape = (img_width, img_height, 3)

dimSz = img_width;



# Pembuatan Arsitektur Deep Learning

base_model = keras.applications.Xception(
    input_shape=(dimSz, dimSz, 1),
    include_top=False)  # Do not include the ImageNet classifier at the top.

base_model.trainable = True

inputs = keras.Input(shape=(dimSz, dimSz, 1))
# We make sure that the base_model is running in inference mode here,
# by passing `training=False`. This is important for fine-tuning, as you will
# learn in a few paragraphs.
x = base_model(inputs, training=True)
# Convert features of shape `base_model.output_shape[1:]` to vectors
x = keras.layers.GlobalAveragePooling2D()(x)
# A Dense classifier with a single unit (binary classification)
outputs = keras.layers.Dense(7)(x)
model = keras.Model(inputs, outputs)






# model = Sequential()
# model.add(Conv2D(64,(3,3),input_shape=(dimSz,dimSz,3),activation='relu'))
# model.add(Conv2D(32,(3,3),input_shape=(dimSz,dimSz,3),activation='relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))

# model.add(Conv2D(32,(3,3), activation="relu"))
# model.add(Conv2D(16,(3,3), activation="relu"))
# model.add(MaxPooling2D(pool_size=(2,2)))

# model.add(Flatten())
# model.add(Dense(activation="relu", units = 2000))
# model.add(Dense(activation="relu", units = 256))
# model.add(Dense(activation="relu", units = 128))
# model.add(Dense(activation="softmax", units = 7))

# model.compile(optimizer ='adam', loss = 'categorical_crossentropy', metrics =['accuracy'])

# os.system('clear') 


# model.compile(loss='categorical_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])
# model.summary()



model.compile(optimizer ='adam', loss = 'categorical_crossentropy', metrics =['accuracy'])
#model.fit(new_dataset, epochs=20, callbacks=..., validation_data=...)

base_model.summary()


# #C:\Users\INKOM06\Pictures\washhand\trainData\huebEr\ds\_fold1\xmodel
model_checkpoint_callback = ModelCheckpoint(rootPath+kfold+"\\xmodel\\hand_best_model_"+kfold+"_"+str(img_width)+".h5", 
     monitor='val_acc', 
     verbose=1,
     save_best_only=True, 
     mode='max', 
     period=1)


# # this is the augmentation configuration we will use for training
# #train_datagen = ImageDataGenerator(
# #    rescale=1. / 255,
# #    shear_range=0.2,
# #    zoom_range=0.2,
# #    horizontal_flip=True)

# # this is the augmentation configuration we will use for testing:
# # only rescaling

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen  = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
     train_data_dir,
     target_size=(img_width, img_height),
     batch_size=batch_size,
     class_mode='categorical')

valid_generator = test_datagen.flow_from_directory(
     validation_data_dir,
     target_size=(img_width, img_height),
     batch_size=batch_size,
     class_mode='categorical')


nb_train_samples = 102
nb_validation_samples = 29


history = model.fit_generator(
             train_generator,
             callbacks=[model_checkpoint_callback],
             steps_per_epoch=nb_train_samples // batch_size,
             epochs=epochs,
             validation_data=valid_generator,
             validation_steps=nb_validation_samples // batch_size,
             verbose = 1)

# #model.save_weights("C:\\Users\\INKOM06\\Pictures\\roadDataset\\bandung\\dataset\\_fold1\\xmodel\\fold1.h5")


import matplotlib.pyplot as plt





print(history.history.keys())

ax = []
for i in range(2):
    ax.append("")

fig, (ax[0:1]) = plt.subplots(1, 2)
fig.suptitle('Plot of model accuracy and loss')

# # summarize history for accuracy
ax[0].plot(history.history['acc'])
ax[0].plot(history.history['val_acc'])
ax[0].set_title("model accuracy")
ax[0].set_ylabel('accuracy')
ax[0].set_xlabel('epoch')
ax[0].legend(['train', 'test'], loc='upper left')

# # summarize history for loss
ax[1].plot(history.history['loss'])
ax[1].plot(history.history['val_loss'])
ax[1].set_title('model loss')
ax[1].set_ylabel('loss')
ax[1].set_xlabel('epoch')
ax[1].legend(['train', 'test'], loc='upper left')

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
