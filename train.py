import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.callbacks import ModelCheckpoint


os.system("cls")



#img_width, img_height = 20, 20
img_width, img_height = 30, 30

kfold = "_fold4"

rootPath = "C:\\Users\\INKOM06\\Pictures\\roadDataset\\bandung\\Purbaleunyi_Highway_1\\ds\\"

#trainFolder = "C:\\Users\\INKOM06\\Pictures\\roadDataset\\bandung\\dataset\\"+kfold+"\\train\\"
#validFolder = "C:\\Users\\INKOM06\\Pictures\\roadDataset\\bandung\\dataset\\"+kfold+"\\valid\\"
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

epochs = 30
batch_size = 15


if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()


#model_checkpoint_callback = ModelCheckpoint("C:\\Users\\INKOM06\\Pictures\\roadDataset\\bandung\\dataset\\"+kfold+"\\xmodel\\best_model_"+kfold+".h5", 
model_checkpoint_callback = ModelCheckpoint(rootPath+kfold+"\\xmodel\\best_model_"+kfold+".h5", 
    monitor='val_acc', 
    verbose=1,
    save_best_only=True, 
    mode='max', 
    period=1)


# this is the augmentation configuration we will use for training
#train_datagen = ImageDataGenerator(
#    rescale=1. / 255,
#    shear_range=0.2,
#    zoom_range=0.2,
#    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen  = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')


nb_train_samples = 1200
nb_validation_samples = 600

#history = model.fit(X_train, y_train,
#          batch_size=batch_size,
#          epochs=epochs,
#          verbose=1,
#          validation_split=0.2)


history = model.fit_generator(
            train_generator,
            callbacks=[model_checkpoint_callback],
            steps_per_epoch=nb_train_samples // batch_size,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=nb_validation_samples // batch_size,
            verbose = 1)

#model.save_weights("C:\\Users\\INKOM06\\Pictures\\roadDataset\\bandung\\dataset\\_fold1\\xmodel\\fold1.h5")

import matplotlib.pyplot as plt





print(history.history.keys())

ax = []
for i in range(2):
    ax.append("")

fig, (ax[0:1]) = plt.subplots(1, 2)
fig.suptitle('Plot of model accuracy and loss')

# summarize history for accuracy
ax[0].plot(history.history['acc'])
ax[0].plot(history.history['val_acc'])
ax[0].set_title("model accuracy")
ax[0].set_ylabel('accuracy')
ax[0].set_xlabel('epoch')
ax[0].legend(['train', 'test'], loc='upper left')

# summarize history for loss
ax[1].plot(history.history['loss'])
ax[1].plot(history.history['val_loss'])
ax[1].set_title('model loss')
ax[1].set_ylabel('loss')
ax[1].set_xlabel('epoch')
ax[1].legend(['train', 'test'], loc='upper left')

plt.show()
