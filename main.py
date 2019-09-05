from time import time
import cv2
import numpy as np
import imutils

faceClassifier = cv2.CascadeClassifier('./models/haarcascade_frontalface_alt.xml')
facemark = cv2.face.createFacemarkLBF()
facemark.loadModel('./models/lbfmodel.yaml')
net = cv2.dnn.readNetFromCaffe('./models/deploy.prototxt.txt',
                               './models/res10_300x300_ssd_iter_140000.caffemodel')
confidenceThreshold = 0.5

cap = cv2.VideoCapture(0)

while cap.isOpened():
    start = time()
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = imutils.resize(frame, width=800)
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    
    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()
    
    # loop over the detections
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        # filter out weak detections
        if confidence < confidenceThreshold:
            continue
        
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        box = box.astype("int")
        (startX, startY, endX, endY) = box
        face = np.array([startX, startY, endX-startX, endY-startY])
        
        landmarks = facemark.fit(frame, np.array([face]))
        for landmark in landmarks[1]:
            for landm in landmark[0]:
                cv2.circle(frame, center=tuple(landm), radius=1, color=(255, 25, 255),
                           thickness=-1)
        cv2.rectangle(frame, (startX, startY), (endX, endY), (126, 65, 64), 1)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    end = time()
    print("fps = %f"%(1/(end-start)), end='\r', flush=True)

cap.release()
cv2.destroyAllWindows()
