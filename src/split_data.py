import os
import random
import shutil


def split_data(image_dir, label_dir, output_dir, split_ratio=0.8):
    # Create train/ and val/ directories for images and labels
    train_img_dir = os.path.join(output_dir, 'train', 'images')
    val_img_dir = os.path.join(output_dir, 'val', 'images')
    train_lbl_dir = os.path.join(output_dir, 'train', 'labels')
    val_lbl_dir = os.path.join(output_dir, 'val', 'labels')

    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(train_lbl_dir, exist_ok=True)
    os.makedirs(val_lbl_dir, exist_ok=True)

    # List all images and shuffle them
    images = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    random.shuffle(images)

    # Calculate the split index
    split_index = int(len(images) * split_ratio)

    # Split images into training and validation sets
    train_images = images[:split_index]
    val_images = images[split_index:]

    # Move or copy the images and corresponding labels
    for img in train_images:
        img_path = os.path.join(image_dir, img)
        label_path = os.path.join(label_dir, img.replace('.jpg', '.txt'))
        shutil.copy(img_path, train_img_dir)  # Copy image to train folder
        shutil.copy(label_path, train_lbl_dir)  # Copy corresponding label to train folder

    for img in val_images:
        img_path = os.path.join(image_dir, img)
        label_path = os.path.join(label_dir, img.replace('.jpg', '.txt'))
        shutil.copy(img_path, val_img_dir)  # Copy image to val folder
        shutil.copy(label_path, val_lbl_dir)  # Copy corresponding label to val folder

    print(f"Training set: {len(train_images)} images")
    print(f"Validation set: {len(val_images)} images")


# Define paths
image_directory = r"C:\Users\a2ham\PycharmProjects\hardhat-detector-app\data\collected_data\images"
label_directory = r"C:\Users\a2ham\PycharmProjects\hardhat-detector-app\data\collected_data\labels"
output_directory = r"C:\Users\a2ham\PycharmProjects\hardhat-detector-app\data\collected_data"

# Split with an 80/20 ratio
split_data(image_directory, label_directory, output_directory, split_ratio=0.8)



