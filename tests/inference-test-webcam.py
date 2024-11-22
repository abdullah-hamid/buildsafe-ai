# Purpose: Perform inference on webcam feed using a trained YOLO model.

import cv2
from ultralytics import YOLO

# Load the trained YOLO model
model = YOLO('../runs/finetune/train2/weights/best.pt') # Replace with model path

# Open a connection to the webcam (0 is the default camera)
cap = cv2.VideoCapture(0)

# Check if the webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Perform inference
    results = model(frame)

    # Process results
    for result in results:
        boxes = result.boxes  # Get the bounding boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]  # Get box coordinates
            conf = box.conf[0]  # Get confidence score
            cls = box.cls[0]  # Get class index
            # Define colors for each class
            colors = {
                0: (0, 0, 255),  # Red for heads (class 0)
                1: (255, 0, 0)  # Blue for hardhats (class 1)
            }
            # Draw bounding box on the frame with appropriate color
            color = colors[int(cls)]
            thickness = 5  # box thickness
            # Draw bounding box on the frame
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
            cv2.putText(frame, f'Class: {int(cls)} Conf: {conf:.2f}', (int(x1), int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the resulting frame
    cv2.imshow("Webcam Feed - press 'q' to quit", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
