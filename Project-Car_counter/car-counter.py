import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *
#webcam
# cap = cv2.VideoCapture(0)
# cap.set(3, 1280)
# cap.set(4, 720)
#video
cap = cv2.VideoCapture("../Videos/cars.mp4")


model = YOLO('../Yolo-Weights/yolov8n.pt')

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush" , "Face Wash"
              ]

mask = cv2.imread("mask.png")
#TRACKING
tracker = Sort(max_age=20,min_hits=3,iou_threshold=0.3)
limits = [405,297,673,297]
count = []
while True:
    success, img = cap.read()
    imgRegion = cv2.bitwise_and(img,mask)
    results = model(imgRegion, stream=True)
    detections=np.empty((0,5))
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1,y1,x2,y2= box.xyxy[0]
            x1, y1, x2, y2 =  int(x1),int(y1),int(x2),int(y2)
            # cv2.rectangle(img,  (x1,y1) ,(x2,y2),(255,0,255),3)
            w,h = x2-x1 , y2-y1
            # cvzone.cornerRect(img,(x1,y1,w,h))
            conf = math.ceil((box.conf[0])*100)/100
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            if currentClass == "car" or currentClass == "truck" or currentClass == "bus" \
                    or currentClass == "motorbike" and conf > 0.3:
                # cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)),
                #                    scale=0.6, thickness=1, offset=3)
                # cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=5)
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)

    for result in resultsTracker:
        x1,y1,x2,y2,id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h))
        cvzone.putTextRect(img, f' {id}', (max(0, x1), max(35, y1)),
                           scale=1, thickness=2, offset=3)
        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        if limits[0]< cx <limits[2] and limits[1] - 20 < cy < limits[1] + 20:
            if count.count(id) == 0:
                count.append(id)

                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255,0), 5)

    cvzone.putTextRect(img, f' Count :{len(count)}', (50, 50))


    cv2.imshow("Webcam",img)
    cv2.waitKey(1)