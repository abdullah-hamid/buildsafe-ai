import cv2
from ultralytics import YOLO
from config import CAMERA_USERNAME, CAMERA_PASSWORD, CAMERA_IP


# Camera settings
username = CAMERA_USERNAME
password = CAMERA_PASSWORD
camera_ip = CAMERA_IP

stream_url = f'rtsp://{username}:{password}@{camera_ip}'

# Load the trained YOLO model
model = YOLO(r"../runs/finetune/train4/weights/best.pt")  # Replace with model path

# Open the stream
cap = cv2.VideoCapture(stream_url)

if not cap.isOpened():
    print("Error: Could not open video stream")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to retrieve frame")
        break

    # Get original frame size
    original_h, original_w = frame.shape[:2]

    # Resize the frame to the expected input size
    resized_frame = cv2.resize(frame, (640, 640))

    # Perform inference
    results = model(resized_frame)

    # Process results
    for result in results:
        boxes = result.boxes  # Get the bounding boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]  # Get box coordinates from resized frame
            conf = box.conf[0]  # Get confidence score
            cls = box.cls[0]  # Get class index

            # Filter by confidence threshold
            if conf < 0.12 or conf > 0.8:  # Adjust threshold as needed
                continue

            # Scale the box coordinates to match the original frame size
            scale_x = original_w / 640  # Scale factor for width
            scale_y = original_h / 640  # Scale factor for height

            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)

            # Define class names instead of numbers
            class_names = {
                0: "no hardhat",  # Class 0 is no hardhat
                1: "hardhat"  # Class 1 is hardhat
            }

            # Get class name from class index
            class_name = class_names[int(cls)]

            # Define colors for each class
            colors = {
                0: (0, 0, 255),  # Red for heads (class 0)
                1: (255, 0, 0)   # Blue for hardhats (class 1)
            }

            # Draw bounding box on the original frame with appropriate color
            color = colors[int(cls)]
            thickness = 3  # Box thickness
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(frame, f'{class_name} Con: {conf:.3f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the frame with bounding boxes
    cv2.imshow('Live Camera Feed', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
