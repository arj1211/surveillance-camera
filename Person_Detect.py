import numpy as np
import cv2
from imutils.object_detection import non_max_suppression

cap = cv2.VideoCapture(0)

#Create HOG desc, import pretrained params
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #winStride determines x, y steps of sliding window for finding HOG, affects
    #Accuracy and speed of algo
    #Scale determines number of lvls in img pyramid, allows for dec size for inc res
    #Could help determine grad dir accurately, at cost of proc time
    #Train on gray-scale for better accuracy/consistency
    (rects, weights) = hog.detectMultiScale(gray, winStride=(4,4),
     padding=(8,8), scale=1.05)
    for(x, y, w, h) in rects:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255), 2)
    
    #Apply NMS, loop over all bounding boxes, contain in arr
    rects = np.array([[x, y, x+w, y+h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

    #Draw rem bb
    for(xA, yA, xB, yB) in pick:
        cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
    
    cv2.imshow("frame", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()
    