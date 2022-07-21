import numpy as np
import cv2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import time

ctime=0
close_flag=0
eye_closed = [1,1]

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
eye_cascadeL = cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')
eye_cascadeR = cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')

model=load_model('eyeclassifier.h5')

cap = cv2.VideoCapture(0)

def predict(img):
    img = cv2.resize(img,(24,24),interpolation = cv2.INTER_AREA)
    img = (image.img_to_array(img))/255
    img = np.expand_dims(img, axis = 0)
    result=model.predict(img)
    if result[0][0]==0:
        return 1
    else:
        return 0

while (1):
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.05, 10)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_grayface = gray[y:y+h, x:x+w]
        roi_colorface = img[y:y+h, x:x+w]
        
        eyesL = eye_cascadeL.detectMultiScale(roi_grayface)
        for i,(ex,ey,ew,eh) in enumerate(eyesL):
            cv2.rectangle(roi_colorface,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            roi_grayeyes = gray[ey:ey+eh, ex:ex+ew]
            roi_coloreyes = img[ey:ey+eh, ex:ex+ew]
            if i<2:
                eye_closed[i] = predict(roi_grayeyes)


        # eyesR = eye_cascadeR.detectMultiScale(roi_grayface)
        # for (ex,ey,ew,eh) in eyesR:
        #     cv2.rectangle(roi_colorface,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        #     roi_grayeyes = gray[ey:ey+eh, ex:ex+ew]
        #     roi_coloreyes = img[ey:ey+eh, ex:ex+ew]

    if all(eye_closed) and close_flag==0:
        ctime = time.time()
        close_flag = 1
    elif close_flag==1 and time.time()-ctime>3 :
        print("Wake up!")
    elif not all(eye_closed):
        ctime=0
        close_flag=0

    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()