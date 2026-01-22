import cv2
import mediapipe as mp
from time import time
from ultralytics import YOLO

# https://docs.ultralytics.com/models/yolo11/#overview
# Load a COCO-pretrained YOLO11n model (medium) -> Other versions https://github.com/ultralytics/assets/releases/
model = YOLO("yolo11m.pt")

# COCO person class id
PERSON_CLASS_ID = 1

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_drawing_styles = mp.solutions.drawing_styles

def process_video(input_video, output_video):
    # choose webcam or input video
    # cap = cv2.VideoCapture(input_video)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error opening video file.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    print(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (960, 540))

    # From https://mediapipe.readthedocs.io/en/latest/solutions/hands.html
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=3,
        min_detection_confidence=0.6
    ) as hands:
        while True:
            t2 = time()
            ret, frame = cap.read()
            if not ret:
                break

            # Make the image RGB instead of GBR
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            t = time()
            results = model(image, stream=True)

            # Make image grayscale for contrast
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            # image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            # image = cv2.resize(image, (960, 540))

            dot_positions = []

            for r in results:
                boxes = r.boxes

                for box in boxes:
                    if box.cls[0] != 0:
                        continue  # Only show people
                    # bounding box
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # convert to int values

                    y1 -= 30  # Make the box 30 px higher on the top, so it includes hands raised high
                    if y1 < 0:
                        y1 = 0

                    # Crop the frame to the bounding box
                    cropped = image[y1:y2, x1:x2]
                    t = time()
                    results = hands.process(cropped)
                    print('hands time:', time() - t)

                    num_hands = 0
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            # Check if the hand is raised above the shoulder
                            wrist_y = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y
                            index_finger_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y
                            shoulder_y = 0.35  # Approximate shoulder height in normalized coordinates

                            if (wrist_y < shoulder_y or index_finger_y < shoulder_y) and num_hands < 1:
                                num_hands += 1
                                # Convert normalized wrist coordinates to pixel coordinates in the full image
                                wrist_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * (x2 - x1)) + x1
                                wrist_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * (y2 - y1)) + y1

                                # Draw a red circle at the wrist position on the full image
                                dot_positions.append((wrist_x, wrist_y))

            frame = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            for dot_position in dot_positions:
                cv2.circle(frame, dot_position, 15, (0, 0, 255), -1)  # Red dot (RGB format)
            frame = cv2.resize(frame, (960, 540))
            for _ in range(int((time() - t2) * 30)):
                out.write(frame)
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_video("ManyStudentsComposite.mp4", "output.mp4")
