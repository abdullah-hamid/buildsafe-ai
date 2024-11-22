# This script cleans up the dataset by deleting empty label files and their corresponding images.
# I used YoloLabel to label the images, and it creates a .txt file for each image with the bounding box coordinates.
# It also creates an empty .txt file if no bounding boxes are present in the image. This and the corresponding image
# file are deleted.

import os


def clean_empty_labels(folder_path):
    # Collect all the txt files in the directory
    txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

    # Initialize counters for deleted files
    deleted_txt_count = 0
    deleted_img_count = 0

    for txt_file in txt_files:
        txt_file_path = os.path.join(folder_path, txt_file)

        # Check if the .txt file is empty
        if os.stat(txt_file_path).st_size == 0:
            # Delete the .txt file
            os.remove(txt_file_path)
            deleted_txt_count += 1

            # Find and delete the corresponding image file (with the same name but .jpg extension)
            img_file = txt_file.replace('.txt', '.jpg')
            img_file_path = os.path.join(folder_path, img_file)

            if os.path.exists(img_file_path):
                os.remove(img_file_path)
                deleted_img_count += 1

    print(f"Deleted {deleted_txt_count} empty txt files and {deleted_img_count} corresponding image files.")


if __name__ == "__main__":
    folder_path = r"C:\Users\a2ham\PycharmProjects\hardhat-detector-app\images\captured_frames_20241108_062427"
    clean_empty_labels(folder_path)

