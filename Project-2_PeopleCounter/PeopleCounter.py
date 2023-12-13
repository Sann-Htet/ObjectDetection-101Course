import cv2
import cvzone
import math
import numpy as np
from ultralytics import YOLO
from sort import *

cap = cv2.VideoCapture("ObjectDetectionYOLO/Videos/people.mp4")

classNames = []
classFile = 'ObjectDetectionYOLO/coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

model = YOLO('ObjectDetectionYOLO/Yolo-weights/yolov8n.pt')

mask = cv2.imread("Project-2_PeopleCounter/mask_resize.png")

# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

limitsUp = [103, 161, 296, 161]
limitsDown = [527, 489, 735, 489]

totalCountUp = []
totalCountDown = []


while True:
    success, img = cap.read()
    imgRegion = cv2.bitwise_and(np.array(img), np.array(mask))

    imgGraphics = cv2.imread("Project-2_PeopleCounter/graphics.png", cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img, imgGraphics, (730, 260))

    results = model(imgRegion, stream=True)
    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            ## Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            w, h = x2 - x1, y2 - y1

            ## Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100

            ## Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            if currentClass == "person" and conf > 0.3:
                # cvzone.cornerRect(img, (x1, y1, w, h), l=10, rt=5)
                # cvzone.putTextRect(img, f'{currentClass} {conf}%', (max(0, x1), max(35, y1)), scale=1, thickness=1, offset=3)
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)

    cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 0, 255), 5)
    cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 0, 255), 5)

    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=10, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(img, f'{int(id)}', (max(0, x1), max(35, y1)), scale=2, thickness=3, offset=10)
        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        if limitsUp[0] < cx < limitsUp[2] and limitsUp[1] - 15 < cy < limitsUp[1] + 15:
            if totalCountUp.count(id) == 0:
                totalCountUp.append(id)
                cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 255, 0), 5)

        if limitsDown[0] < cx < limitsDown[2] and limitsDown[1] - 15 < cy < limitsDown[1] + 15:
            if totalCountDown.count(id) == 0:
                totalCountDown.append(id)
                cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 255, 0), 5)
            
    # # cvzone.putTextRect(img, f'Count: {len(totalCount)}', (50, 50))
        
    cv2.putText(img, str(len(totalCountUp)), (929, 345), cv2.FONT_HERSHEY_PLAIN, 5, (139, 195, 75), 7)
    cv2.putText(img, str(len(totalCountDown)), (1191, 345), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 230), 7)


    cv2.imshow("Image", img)
    # cv2.imshow("Image Region", imgRegion)

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()