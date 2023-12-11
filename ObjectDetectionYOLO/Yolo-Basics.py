from ultralytics import YOLO
import cv2

model = YOLO('ObjectDetectionYOLO/Yolo-weights/yolov8n.pt')
results = model("ObjectDetectionYOLO/images/1.jpg", show=True)
cv2.waitKey(0)