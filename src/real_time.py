from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import imutils
import cv2

detection_model_path = '../trained_models/detection_models/haarcascade_frontalface_default.xml'
emotion_model_path = '../trained_models/fer2013_XCEPTION.54-0.66.hdf5'
emotions = ["angry" ,"disgust","scared", "happy", "sad", "surprised", "neutral"]
alpha = 0.04

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


cv2.namedWindow('WebCam')
camera = cv2.VideoCapture(0)
while True:
    frame = imutils.resize(camera.read()[1],width=1280, height=960)
    frameClone = frame.copy()
    output = frame.copy()    
    gray, faces = detect_faces(frame)
    if len(faces) > 0:
        faces = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))
        for face in faces:
            (fX, fY, fW, fH) = face
            roi = cv2.resize(gray[fY:fY + fH, fX:fX + fW], (64, 64))
            roi = roi.astype("float") / 255.0
            roi = np.expand_dims(img_to_array(roi), axis=0)
            preds = classifier.predict(roi)[0]
            emotion_probability = np.max(preds)
            label = emotions[preds.argmax()]
            for (i, (emotion, prob)) in enumerate(zip(emotions, preds)):
                        print("%s - %s" % (emotion, prob)) # Collect 7 data types and find their means. 
                        emotion_label, means_emotion = get_means_emotions(emotion, prob)
                        text = "{}: {:.2f}%".format(emotion, prob * 100)
                        w = int(prob * 200)
                        cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH + 125), (0, 0, 0), -1)
                        cv2.addWeighted(frameClone, alpha, output, 1 - alpha, 0, output)

                        cv2.rectangle(output, (fX, (i * 15) + fY + fH + 10), ( fX + w, (i * 15) + fY + fH + 20), (0, 0, 255), -1)

                        cv2.putText(output, text, (fX, (i * 15) + fY + fH + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
                        cv2.putText(frameClone, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)        
    else: continue

    cv2.imshow('WebCam', output)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    else: continue
camera.release()
cv2.destroyAllWindows()
