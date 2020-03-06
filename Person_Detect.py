import numpy as np
import cv2
import datetime
from imutils.object_detection import non_max_suppression

cap = cv2.VideoCapture(0)
hog = cv2.HOGDescriptor()#Create HOG desc, import pretrained params
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0)
    cl1 = clahe.apply(gray) #Apply CLAHE for consistent lighting b/w diff cam env
   
    #winStride determines x, y steps of sliding window for finding HOG, affects accuracy, speed
    #Scale determines number of lvls img pyramid, allows dec size for inc res, could help determine grad dir accurately, at cost of proc time
    (rects, weights) = hog.detectMultiScale(cl1, winStride=(4,4),
     padding=(8,8), scale=1.05)
    
    #Full body person detection
    #Apply NMS, loop over all bounding boxes, contain in arr
    rects = np.array([[x, y, x+w, y+h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.4) 
    person = np.array(pick) #create np arr to ref size

    
    #Draw remaining bounding boxes
    for(xA, yA, xB, yB) in pick:
        cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
        
    #Face detection
    #CascadeClassifier trained of fac recognition model
    objects = cv2.CascadeClassifier("facial_recognition_model.xml").detectMultiScale(
                cl1,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(20, 20),
                flags=cv2.CASCADE_SCALE_IMAGE)
    for (x, y, w, h) in objects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2) #cascadeclassifier emptiness via .empty() method
    faces = np.array(objects) #convert to np arr for comparison

    #Check that both person & face detection are working, then show text
    if(person.size != 0 and faces.size != 0):
        frame = cv2.putText(frame, "Person Detected", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    #Set up current date/time    
    date = datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    #(frame, text, pos, font, thickness, colour, thickness, line-type)    
    frame = cv2.putText(frame, date, (265, 465), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    
    cv2.imshow("Live Feed", frame)
    key = cv2.waitKey(1)
    if key == 27: #esc to exit
        break
    
cap.release()
cv2.destroyAllWindows()
    