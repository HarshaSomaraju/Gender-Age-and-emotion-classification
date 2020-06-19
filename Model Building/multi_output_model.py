# -*- coding: utf-8 -*-

import keras
from keras.models import Sequential,Model,Input
from keras.layers import Conv2D,MaxPooling2D,BatchNormalization,Dropout,Dense,Flatten

inputShape=(64,64,3)

model= Sequential()
model.add(Conv2D(32,(3,3),padding="same",input_shape=inputShape,activation='relu'))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Dropout(0.25))

model.add(Conv2D(32,(1,1)))
model.add(BatchNormalization(axis=-1))

model.add(Conv2D(64,(3,3),padding="same",activation='relu'))
model.add(BatchNormalization(axis=-1))

model.add(Conv2D(64,(1,1)))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128,(3,3),padding="same",activation='relu'))
model.add(BatchNormalization(axis=-1))

model.add(Conv2D(128,(1,1)))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(1024,activation='relu'))
model.add(BatchNormalization(axis=-1))
model.add(Dropout(0.25))
x = model.get_output_at(-1)

y1 = Dense(128,activation='relu')(x)
y1 = Dropout(0.25)(y1)

y2 = Dense(128,activation='relu')(x)
y2 = Dropout(0.25)(y2)

y1 = Dense(2,activation='softmax',name='gender')(y1)
y2 = Dense(13,activation='softmax',name='age')(y2)

mod = Model(inputs = model.get_input_at(0), outputs = [y1,y2] )

mod.summary()


mod.save("../2-class-base-dropouts.h5")

"""Emotions Model"""

model = Sequential()
model.add(Conv2D(32,(3,3),padding="same",input_shape=inputShape,activation='relu'))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Dropout(0.25))

model.add(Conv2D(32,(1,1)))
model.add(BatchNormalization(axis=-1))

model.add(Conv2D(64,(3,3),padding="same",activation='relu'))
model.add(BatchNormalization(axis=-1))
model.add(Conv2D(64,(1,1)))
model.add(BatchNormalization(axis=-1))

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128,(3,3),padding="same",activation='relu'))
model.add(BatchNormalization(axis=-1))
model.add(Conv2D(128,(1,1)))
model.add(BatchNormalization(axis=-1))

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024,activation='relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization(axis=-1))
model.add(Dense(2,activation='softmax'))

model.save('../Emotion_Base_model.h5')

