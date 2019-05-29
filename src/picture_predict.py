from keras.preprocessing.image import img_to_array
from keras.models import load_model
from PIL import Image
import numpy as np
import imutils
import cv2
import csv
import glob
import os


detection_model_path = '../trained_models/detection_models/haarcascade_frontalface_default.xml'
emotion_model_path = '../trained_models/emotion_models/fer2013_XCEPTION.54-0.66.hdf5'
emotions = ["angry" ,"disgust","scared", "happy", "sad", "surprised", "neutral"]
alpha = 0.04
alp = 0.6
face_detection = cv2.CascadeClassifier(detection_model_path)
classifier = load_model(emotion_model_path, compile=False)


def detect_faces(frame):
    """ This function find faces from webcam. """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30), flags=cv2.CASCADE_SCALE_IMAGE)
    return gray, faces

def get_means_emotions(e_label=None, m_emotion=None):
    """ This function will get means from emotion values. """
    return e_label, m_emotion
a=input("path gir")
img_list=os.listdir(a)
image_list = []
for i in img_list:
    try:
        im=Image.open(a + "/" + i)
        image_list.append(i)
    except IOError:
        pass

cv2.namedWindow('WebCam')
for imgL in image_list:
    frame = cv2.imread(a+'/'+imgL);
    frameClone = frame
    output = frame
    try:
        gray, faces = detect_faces(frame)
        face_detected = False 
        if len(faces) > 0:
            faces = sorted(faces, reverse=True, key=lambda x: (x[2]) * (x[3]))
            for face in faces:
                (fX, fY, fW, fH) = face
                roi = cv2.resize(gray[fY:fY + fH, fX:fX + fW], (64, 64))
                roi = roi.astype("float") / 255.0
                roi = np.expand_dims(img_to_array(roi), axis=0)
                preds_o = classifier.predict(roi)[0]
                if (face_detected==False):
                    emotion_probability_mean = preds_o
                    face_detected = True

                emotion_probability_mean = emotion_probability_mean * (1-alp)+ preds_o * alp
                preds = emotion_probability_mean
                emotion_probability = np.max(preds)

                label = emotions[preds.argmax()]

                writeAsList=[] 
                writeAsList.append(imgL)
                writeAsList.append(label)
                writeAsList.append(max(preds))

                print(writeAsList)

                with open(a +'/preds.csv', 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow(writeAsList)
    except:
        pass
