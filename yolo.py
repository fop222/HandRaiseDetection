import cv2
from ultralytics import YOLO

# Load a COCO-pretrained YOLO11n model
model = YOLO("yolo11m.pt")

input_video = "ManyStudentsComposite.mp4"
cap = cv2.VideoCapture(input_video)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes

        for box in boxes:
            if box.cls[0] != 0:
                continue  # Only show people
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # put box in cam
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
            confidence = int((box.conf[0]*100))/100
            print("Confidence --->",confidence)

    cv2.imshow('Output', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if cv2.waitKey(1) == ord('q'):
        break

results = model()
