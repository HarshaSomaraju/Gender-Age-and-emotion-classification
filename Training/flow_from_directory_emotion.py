# -*- coding: utf-8 -*-

import keras
import numpy as np
import cv2

path = '../Smile_Dataset'

import glob

len(glob.glob(path+'/*/*'))

from keras.models import load_model

model = load_model('../Emotion_Base_model.h5')

from keras.preprocessing.image import ImageDataGenerator

datagen=ImageDataGenerator(rescale=1./255.,validation_split=0.25)



x_train=datagen.flow_from_directory(path,target_size=(64,64),batch_size=32,class_mode='categorical',shuffle='true',subset='training')

x_train.class_indices

x_validate=datagen.flow_from_directory(path,target_size=(64,64),batch_size=32,class_mode='categorical',shuffle='true',subset='validation')

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

history=model.fit_generator(generator=x_train,steps_per_epoch=x_train.samples//x_train.batch_size, epochs = 100, validation_data= x_validate,validation_steps= x_validate.samples//x_validate.batch_size)

model.save('../Emotion_Base_model.h5')

import matplotlib.pyplot as plt
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

model.save('../trained_emotion_model.h5')
