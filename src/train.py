import os
import json
from ultralytics import YOLO


def main():
    # Create a models directory and get the path to the settings file
    root_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # This gets the parent directory of the current file's directory
    models_directory = os.path.join(root_directory, 'models')  # Combine to form the path to models directory
    config_directory = os.path.join(root_directory, 'config')  # Combine to form the path to config directory
    settings_path = os.path.join(config_directory, 'settings.json')

    # Create the models directory if it does not exist
    os.makedirs(models_directory, exist_ok=True)

    # Define path to the model weights
    model_weights_path = os.path.join(models_directory, 'yolov8m.pt')

    # Load a YOLOv8 model and save it to the models directory (or download if necessary)
    if not os.path.exists(model_weights_path):
        print(f"Model weights not found at {model_weights_path}, downloading...")
        model = YOLO('yolov8m')  # Automatically downloads the model if not available
    else:
        model = YOLO(model_weights_path)  # Load the model from local weights

    # Correct the dataset path (remove unnecessary parent directory)
    data_yaml_path = os.path.join(root_directory, 'data', 'roboflow', 'data.yaml')

    # Ensure the dataset file exists before training
    if not os.path.exists(data_yaml_path):
        raise FileNotFoundError(f"Dataset YAML file not found at: {data_yaml_path}")

    # Train the model on a Roboflow dataset
    model.train(data=data_yaml_path, imgsz=640, epochs=150, patience=15)

if __name__ == '__main__':
    main()
