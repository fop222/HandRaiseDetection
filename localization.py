import cv2
import torch
import torchvision
from torchvision.transforms import functional as F

# Load pre-trained Faster R-CNN model on COCO.
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# COCO person class id
PERSON_CLASS_ID = 1

def process_video(input_video, output_video):
    cap = cv2.VideoCapture(input_video)
    # cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error opening video file.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Parameters for cropped video
    crop_size = 224  # square size for cropped output
    cropped_video_writer = None
    cropped_output_video = "cropped_output.mp4"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # frame = cv2.resize(frame, (320, 180))

        # Preprocess frame: convert from BGR to RGB, then tensor.
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_tensor = F.to_tensor(image)

        with torch.no_grad():
            outputs = model([image_tensor])
        # Filter detections for persons with score >= 0.8.
        boxes = outputs[0]['boxes']
        labels = outputs[0]['labels']
        scores = outputs[0]['scores']
        threshold = 0.8
        # for box, label, score in zip(boxes, labels, scores):
        box, label, score = next(zip(boxes, labels, scores))
        if score >= threshold and label.item() == PERSON_CLASS_ID:
            x1, y1, x2, y2 = box.int().tolist()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Crop the frame to the bounding box
            cropped = frame[y1:y2, x1:x2]
            # Resize to square
            cropped_resized = cv2.resize(cropped, (crop_size, crop_size))

            # Initialize cropped video writer if not already done
            if cropped_video_writer is None:
                fourcc_crop = cv2.VideoWriter_fourcc(*'mp4v')
                cropped_video_writer = cv2.VideoWriter(
                    cropped_output_video, fourcc_crop, fps, (crop_size, crop_size)
                )
            # Write cropped frame to cropped video
            cropped_video_writer.write(cropped_resized)

        # out.write(frame)
        # Optional: display the frame in a window.
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    if cropped_video_writer is not None:
        cropped_video_writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_video("ManyStudentsComposite.mp4", "output.mp4")
