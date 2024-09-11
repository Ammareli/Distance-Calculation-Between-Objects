import cv2 as cv
import mediapipe as mp
from ultralytics import YOLO
import math


model = YOLO("yolov8n.pt")
names = model.model.names

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hand = mp.solutions.hands 

p = mp.solutions.hands.Hands()


# object classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]


def process(frame):
    result = p.process(frame)
    if result.multi_hand_landmarks:
            for land_marks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame,land_marks,mp_hand.HAND_CONNECTIONS )
            
    return frame    

def detect_object(frame):
    results = model(frame, stream=True)
    for r in results:
        boxes = r.boxes

          
        for box in boxes:
                # bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

                # confidence
                confidence = math.ceil((box.conf[0]*100))/100
                print("Confidence --->",confidence)

                if confidence > 0.80:
                     
                    # put box in cam
                    cv.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

                    

                    # class name
                    cls = int(box.cls[0])
                    print("Class name -->", classNames[cls])

                    # object details
                    org = [x1, y1]
                    font = cv.FONT_HERSHEY_SIMPLEX
                    fontScale = 1
                    color = (255, 0, 0)
                    thickness = 2

                    cv.putText(frame, classNames[cls], org, font, fontScale, color, thickness)
                    
    return frame

    





