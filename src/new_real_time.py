from keras.preprocessing.image import img_to_array
from keras.models import load_model
from queue import Queue, Full
import numpy as np
import imutils
import pandas as pd
import cv2

detection_model_path = '../trained_models/detection_models/haarcascade_frontalface_default.xml'
emotion_model_path = '../trained_models/emotion_models/fer2013_XCEPTION.54-0.66.hdf5'
emotions = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]
alpha = 0.04
alp = 0.6
face_detection = cv2.CascadeClassifier(detection_model_path)
classifier = load_model(emotion_model_path, compile=False)
SECOND = 10
max_qlength = 60 * SECOND
frame_queue = Queue(maxsize=max_qlength)


def detect_faces(frame):
    """ This function find faces from webcam. """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                            minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    return gray, faces


def get_all_queue_result(preds_list):
    """ This function average emotions. """
    df = pd.DataFrame(preds_list)
    return df.mean(axis=0)


def get_means_emotions(predictions=None):
    """ This function will get means from emotion values. """
    try:
        frame_queue.put_nowait(predictions)
    except Full:
        frame_queue.get()
        frame_queue.put_nowait(predictions)
    return get_all_queue_result(list(frame_queue.queue))


cv2.namedWindow('WebCam')
camera = cv2.VideoCapture(0)
while True:
    frame = imutils.resize(camera.read()[1], width=850, height=850)
    frameClone = frame.copy()
    output = frame.copy()    
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
            if not face_detected:
                emotion_probability_mean = get_means_emotions(preds_o)
                face_detected = True

            emotion_probability_mean = emotion_probability_mean * (1-alp) + preds_o * alp
            preds = emotion_probability_mean
            emotion_probability = np.max(preds)

            label = emotions[preds.argmax()]
            for (i, (emotion, prob)) in enumerate(zip(emotions, preds)):
                        print("%s - %s" % (emotion, prob))  # Collect 7 data types and find their means.
                        emotion_label, means_emotion = emotion, prob
                        text = "{}: {:.2f}%".format(emotion, prob * 100)
                        w = int(prob * 200)
                        cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH + 125), (0, 0, 0), -1)
                        cv2.addWeighted(frameClone, alpha, output, 1 - alpha, 0, output)

                        cv2.rectangle(output, (fX, (i * 15) + fY + fH + 10), (fX + w, (i * 15) + fY + fH + 20), (0, 0, 255), -1)

                        cv2.putText(output, text, (fX, (i * 15) + fY + fH + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
                        cv2.putText(frameClone, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)        
    else: continue

    cv2.imshow('WebCam', output)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    else: continue
camera.release()
cv2.destroyAllWindows()
