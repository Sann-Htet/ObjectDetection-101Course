import cv2
import cvzone
import math
import numpy as np
from ultralytics import YOLO

cap = cv2.VideoCapture("ObjectDetectionYOLO/Videos/cars.mp4")

classNames = []
classFile = 'ObjectDetectionYOLO/coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

model = YOLO('ObjectDetectionYOLO/Yolo-weights/yolov8n.pt')

mask = cv2.imread("Project-1_CarCounter/mask_resize.png")

while True:
    success, img = cap.read()
    imgRegion = cv2.bitwise_and(np.array(img), np.array(mask))

    results = model(img, stream=True)

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
            if currentClass == "car" or currentClass == "truck" or currentClass == "bus" or currentClass == "motorbike" and conf > 0.3:
                cvzone.cornerRect(img, (x1, y1, w, h), l=10)
                cvzone.putTextRect(img, f'{currentClass} {conf}%', (max(0, x1), max(35, y1)), scale=1, thickness=1, offset=3)

    cv2.imshow("Image", img)
    cv2.imshow("Image Region", imgRegion)

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()