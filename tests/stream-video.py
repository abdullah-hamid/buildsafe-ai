# Purpose: Stream video from an IP camera using OpenCV.

import cv2
from config import CAMERA_USERNAME, CAMERA_PASSWORD, CAMERA_IP


# Camera settings
username = CAMERA_USERNAME
password = CAMERA_PASSWORD
camera_ip = CAMERA_IP

stream_url = f'rtsp://{username}:{password}@{camera_ip}'

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

    # Resize the frame to the expected input size
    resized_frame = cv2.resize(frame, (640, 640))

    # Display the frame
    cv2.imshow('Live Camera Feed - press q to quit', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


