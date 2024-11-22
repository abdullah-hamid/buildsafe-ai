import cv2
import random
import os
from PIL import Image, ImageEnhance
import numpy as np  # Make sure numpy is imported


# Function to apply random rotation to the image
def random_rotation(image):
    angle = random.randint(-15, 15)  # Rotate image by a random angle between -15 and 15 degrees
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated


# Function to apply random brightness changes
def random_brightness(image):
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert from OpenCV to PIL format
    enhancer = ImageEnhance.Brightness(pil_img)  # Enhance brightness
    factor = random.uniform(0.7, 1.3)  # Random brightness factor between 0.7 (darker) and 1.3 (brighter)
    brightened = enhancer.enhance(factor)  # Apply the brightness factor
    return cv2.cvtColor(np.array(brightened), cv2.COLOR_RGB2BGR)  # Convert back to OpenCV format


# Function to apply random horizontal flip
def random_flip(image):
    if random.random() > 0.5:
        return cv2.flip(image, 1)  # Flip the image horizontally
    return image


# Function to apply a series of augmentations
def augment_image(image):
    image = random_flip(image)  # Apply random flip
    image = random_rotation(image)  # Apply random rotation
    image = random_brightness(image)  # Apply random brightness change
    return image


# Main function to augment images
def main():
    # Set the input and output directories
    input_dir = r"C:\Users\a2ham\PycharmProjects\hardhat-detector-app\data\collected data\Images"  # Directory containing original images
    output_dir = r"C:\Users\a2ham\PycharmProjects\hardhat-detector-app\images\Augmented Images" # Directory to save augmented images
    os.makedirs(output_dir, exist_ok=True)  # Create the output directory if it doesn't exist

    # Get a list of image files in the input directory
    images = [f for f in os.listdir(input_dir) if f.endswith('.jpg')]
    num_to_augment = 100  # Number of random images to augment
    random_images = random.sample(images, min(num_to_augment, len(images)))  # Randomly select images

    # Loop through the selected images and apply augmentation
    for i, image_file in enumerate(random_images):
        img_path = os.path.join(input_dir, image_file)  # Path to the current image
        image = cv2.imread(img_path)  # Read the image using OpenCV
        augmented_image = augment_image(image)  # Apply augmentations

        # Set the output file path and save the augmented image
        output_path = os.path.join(output_dir, f'augmented_{i+1}.jpg')
        cv2.imwrite(output_path, augmented_image)
        print(f"Saved augmented image: {output_path}")  # Print the output file path


# Run the main function if this script is executed directly
if __name__ == '__main__':
    main()
