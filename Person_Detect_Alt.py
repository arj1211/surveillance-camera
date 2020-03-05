import cv2
import time
import numpy as np
           
cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #Multiscale on pyramid using CascadeClassifier, pretrained model
    objects = cv2.CascadeClassifier("fullbody_recognition_model.xml").detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(20, 20),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
    #Define minSize as smallest possible window
    #Cascade classifier uses sliding window approach, after each iteration
    #Window reduces in size until breakpoint, used to quickly elim neg, results many false pos

        # Draw a rectangle around the objects
    for (x, y, w, h) in objects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
    
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

        