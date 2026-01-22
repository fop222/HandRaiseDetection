import cv2
import mediapipe as mp

def main():
    # Path to your video file.
    video_path = "cropped_output.mp4"
    cap = cv2.VideoCapture(video_path)
    # cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error opening video file")
        return

    # Initialize MediaPipe drawing utilities and Hands module instead of Pose
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    mp_drawing_styles = mp.solutions.drawing_styles

    # Initialize the HOG-based person (pedestrian) detector.
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # Since we are processing individual ROIs (cropped images of persons),
    # we use static_image_mode=True.
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5
    ) as hands_roi:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                # End of video
                break

            # For person detection, convert the frame to grayscale.
            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect people in the frame.
            # rects, weights = hog.detectMultiScale(
            #     gray,
            #     winStride=(4, 4),
            #     padding=(8, 8),
            #     scale=1.2
            # )

            # Process each detected person.
            # for (x, y, w, h) in rects:
            #     # Adjust the bounding box to ensure it lies within the frame.
            #     x = max(x, 0)
            #     y = max(y, 0)
            #     # Crop the region of interest (ROI) from the frame.
            #     roi = frame[y:y + h, x:x + w].copy()
            #     if roi.size == 0:
            #         continue

            #     # Convert ROI to RGB as MediaPipe expects RGB input.
            #     roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the ROI with MediaPipe Hands.
            result = hands_roi.process(frame_rgb)

            # If hand landmarks are detected, draw them on the ROI.
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame_rgb,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
                    print(hand_landmarks)

                # Annotate the bounding box.
                # cv2.rectangle(frame_rgb, (0, 0), (w - 1, h - 1), (255, 0, 0), 2)

                # Replace the ROI in the main frame with the annotated ROI.
                # frame[y:y + h, x:x + w] = roi

            # Display the frame with all drawn hands.
            cv2.imshow("Hand Detection", frame_rgb)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    # Clean up.
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
