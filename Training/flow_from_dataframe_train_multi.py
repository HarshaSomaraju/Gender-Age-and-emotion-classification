# -*- coding: utf-8 -*-

import numpy as np
import cv2
import pandas as pd
import zipfile
path_to_zip_file='../Manual Dataset.zip'
path='../'
with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
    zip_ref.extractall(path)

import glob
a = glob.glob(path+'/Manual DataSet/*/*/*')
len(a)

int_to_gen = {0: 'female',1: 'male'}
gen_to_int = {'Female':0,'Male':1}
int_to_age = {
    0: '(0-5)',
    1: '(6-10)',
    2: '(11-15)',
    3:'(16-20)',
    4:'(21-25)',
    5:'(26-30)',
    6:'(31-35)',
    7:'(36-40)',
    8:'(41-45)',
    9:'(46-50)',
    10:'(51-55)',
    11:'(56-60)',
    12:'(61-100)'
}
age_to_int = {
    '1-5': 0,
    '6-10': 1 ,
    '11-15': 2,
    '16-20': 3,
    '21-25': 4,
    '26-30': 5,
    '31-35': 6,
    '36-40': 7,
    '41-45': 8,
    '46-50': 9,
    '51-55': 10,
    '56-60': 11,
    '61-100': 12
}

gen=[]
age=[]
for da in a:
  gen.append(gen_to_int[da.split('\\')[-3]])
  age.append(age_to_int[da.split('\\')[-2].split(' ')[0]])

from keras.utils import to_categorical

age=to_categorical(age,dtype='int')

gen=to_categorical(gen,dtype='int')

df = pd.DataFrame({'path':a,'gender':gen.tolist(),'age':age.tolist()})

df.head()

from keras.preprocessing.image import ImageDataGenerator

datagen=ImageDataGenerator(rescale=1./255.,validation_split=0.30)

train_generator=datagen.flow_from_dataframe(
dataframe=df,
directory=None,
x_col='path',
y_col=['gender','age'],
subset="training",
batch_size=32,
seed=42,
shuffle=True,
class_mode="multi_output",
target_size=(64,64))

validation_generator=datagen.flow_from_dataframe(
dataframe=df,
directory=None,
x_col='path',
y_col=['gender','age'],
subset="validation",
batch_size=32,
seed=42,
shuffle=True,
class_mode="multi_output",
target_size=(64,64))

from keras.models import load_model

model = load_model('../2-class-base-dropouts.h5')

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

history=model.fit_generator(generator=validation_generator,steps_per_epoch=validation_generator.samples//validation_generator.batch_size,epochs=40,validation_data=train_generator,validation_steps=train_generator.samples//train_generator.batch_size)

model.save('../2-Trained_model.h5')
