from ultralytics import YOLO


if __name__ == "__main__":
    # Load the pre-trained model with the weights from previous fine-tuning
    model = YOLO('../runs/detect/train3/weights/best.pt')

    # Start training/fine-tuning with new dataset
    model.train(
        data='C:/Users/a2ham/PycharmProjects/hardhat-detector-app/data/collected_data/data.yaml',  # Dataset YAML
        epochs=500,
        #batch=16,
        imgsz=640,
        patience= 25,  # Early stopping patience
        project='../runs/finetune',  # Save fine-tuning results in the 'finetune' folder
    )
