# -*- coding: utf-8 -*-
"""
Created on Mon May  3 13:27:11 2021

@author: DELL
"""

import cv2

# Load and create faceCascade Classifier
#facePath = os.path.dirname(cv2.__file__)+"haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#eyePath = os.path.dirname(cv2.__file__)+"haarcascade_eye.xml"
#eyeCascade = cv2.CascadeClassifier("haarcascade_eye")

out = cv2.VideoWriter('faceEyeDetection.mp4', -1, 20.0, (640,480))
video = cv2.VideoCapture(0)

while True:
    try:
        ret, frames = video.read()
        gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)

        # Face Detection
        faces = faceCascade.detectMultiScale(gray)
        for (x, y, w, h) in faces: cv2.rectangle(frames, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Eye Detection
        #eyes = eyeCascade.detectMultiScale(gray)
        #for (x, y, w, h) in eyes: cv2.rectangle(frames, (x, y), (x + w, y + h), (0, 165, 255), 2)

        # Save and Show the Video
        out.write(frames)
        cv2.imshow('WebCam', frames)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

    except Exception as e:
        print(str(e))
        break

video.release()
out.release()
cv2.destroyAllWindows()
