# Purpose: Test the YOLO model on a single image (i.e. frame) and display the results.

import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Load the trained YOLO model
model = YOLO(r"../runs/detect/train/weights/best.pt")  # Replace with model path

# Load the image
image_path = r"C:\Users\a2ham\Downloads\P_Mazzera-0105-1200x1558.jpg" # Replace with image path
image = cv2.imread(image_path)

# Perform inference
results = model(image)

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
        # Draw bounding box on the image with appropriate color
        color = colors[int(cls)]
        thickness = 10  # box thickness
        # Draw bounding box on the image
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color,thickness)
        cv2.putText(image, f'Class: {int(cls)} Conf: {conf:.2f}', (int(x1), int(y1)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Convert BGR to RGB for displaying with Matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Display the image with detected objects
plt.imshow(image_rgb)
plt.axis('off')  # Hide axes
plt.show()
