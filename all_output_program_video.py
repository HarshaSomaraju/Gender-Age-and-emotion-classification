# -*- coding: utf-8 -*-

import keras

import cv2,numpy as np
from PIL import Image
from keras.preprocessing.image import img_to_array, load_img

from keras.models import load_model

a = load_model('2-Trained_model.h5')
b = load_model('trained_emotion_model.h5')

int_to_gen = {0: 'female',1: 'male'}
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
int_to_emotion = {
    0: 'No Smile',
    1: 'Smile'
}

facer = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    print("Could not open webcam")
    exit()
    
while webcam.isOpened():


    status, frame = webcam.read()

    if not status:
        print("Could not read frame")
        exit()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facer.detectMultiScale(
      gray,
      scaleFactor= 1.1,
      minNeighbors= 5,
      minSize=(10, 10)
    )

    image = (frame[...,::-1].astype(np.float32)) / 255.0
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        face_crop = np.copy(image[y:y+h,x:x+w])
        face_crop = cv2.resize(face_crop,(64,64))
        if face_crop.any():
            i = face_crop
            ans=a.predict(np.expand_dims(i,0))
            ans_1=b.predict(np.expand_dims(i,0))
            gender=int_to_gen[ans[0].argmax()]
            age=int_to_age[ans[1].argmax()]
            emotion=int_to_emotion[ans_1[0].argmax()]
            cv2.putText(frame, gender+' '+age+' '+emotion, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0,255,0), 2)
    try:
        if not faces:
            i = face_crop
            ans=a.predict(np.expand_dims(i,0))
            ans_1=b.predict(np.expand_dims(i,0))
            gender=int_to_gen[ans[0].argmax()]
            age=int_to_age[ans[1].argmax()]
            emotion=int_to_emotion[ans_1[0].argmax()]
            cv2.putText(frame, gender+' '+age+' '+emotion, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0,255,0), 2)
    except:
        pass
        
    cv2.imshow('image',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
webcam.release()
cv2.destroyAllWindows()

