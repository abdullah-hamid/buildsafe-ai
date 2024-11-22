import cv2
import os
import time
from datetime import datetime
from config import CAMERA_USERNAME, CAMERA_PASSWORD, CAMERA_IP

def main():
    # Camera settings
    username = CAMERA_USERNAME
    password = CAMERA_PASSWORD
    camera_ip = CAMERA_IP

    stream_url = f'rtsp://{username}:{password}@{camera_ip}'

    # Create a unique subdirectory for each session using a timestamp
    session_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    images_directory = os.path.join(os.path.dirname(__file__), '..', 'images', f'captured_frames_{session_timestamp}')
    os.makedirs(images_directory, exist_ok=True)

    # Open the camera stream
    cap = cv2.VideoCapture(stream_url)

    if not cap.isOpened():
        print("Error: Could not open video stream")
        return

    frame_interval = 3  # Capture every 3 seconds
    last_capture_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to retrieve frame")
            time.sleep(5)  # Wait before retrying
            continue

        # Check if it's time to capture a new frame (i.e. ignore buffered frames)
        current_time = time.time()
        if current_time - last_capture_time >= frame_interval:
            # Save the frame
            frame_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            frame_filename = os.path.join(images_directory, f'frame_{frame_timestamp}.jpg')
            cv2.imwrite(frame_filename, frame)
            print(f"Captured frame: {frame_filename}")
            last_capture_time = current_time

if __name__ == '__main__':
    main()
