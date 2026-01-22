import cv2
import mediapipe as mp

def main():
    # Path to your video file.
    video_path = "ManyStudentsComposite.mp4"
    # cap = cv2.VideoCapture(video_path)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error opening video file")
        return

    # Initialize MediaPipe drawing utilities and Pose module.
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    # Initialize the HOG-based person (pedestrian) detector.
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # Since we are processing individual ROIs (cropped images of persons),
    # we use static_image_mode=True.
    with mp_pose.Pose(
        static_image_mode=True,
        min_detection_confidence=0.65
    ) as pose_roi:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                # End of video
                break

            # Resize frame for faster processing.
            # frame = cv2.resize(frame, (640, 360))

            # Optionally, you might resize the frame for faster processing.
            # For person detection, convert the frame to grayscale.
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect people in the frame.
            # The detectMultiScale method returns bounding boxes for people.
            rects, weights = hog.detectMultiScale(
                gray,
                winStride=(4, 4),
                padding=(8, 8),
                scale=1.2  # changed from 1.05 for speed
            )

            # (Optional) Draw bounding boxes for detected people.
            # for (x, y, w, h) in rects:
            #     cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Process each detected person.
            for (x, y, w, h) in rects:
                # Adjust the bounding box to ensure it lies within the frame.
                x = max(x, 0)
                y = max(y, 0)
                # Crop the region of interest (ROI) from the frame.
                roi = frame[y:y + h, x:x + w].copy()
                if roi.size == 0:
                    continue

                # Convert ROI to RGB as MediaPipe expects RGB input.
                roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

                # Process the ROI with MediaPipe Pose.
                result = pose_roi.process(roi_rgb)

                # If pose landmarks are detected, draw them on the ROI.
                if result.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        roi,
                        result.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                    )
                    # Optionally, you can also annotate the bounding box.
                    cv2.rectangle(roi, (0, 0), (w - 1, h - 1), (255, 0, 0), 2)

                    # Replace the ROI in the main frame with the annotated ROI.
                    frame[y:y + h, x:x + w] = roi

            # Display the frame with all drawn poses.
            cv2.imshow("Multi-Person Pose", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    # Clean up.
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
