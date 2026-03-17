import os
import random
import shutil

def move_to_validation(image_folder_train, image_folder_val, label_folder_train, label_folder_val, validation_ratio):
    # Moves a percentage of random images and labels from the training folders to validation folders 

    # Get list of files in the train folders
    train_image_files = os.listdir(image_folder_train)
    
    # Calculate the number of files to move to validation set
    num_validation = int(len(train_image_files) * validation_ratio)

    # Randomly select files for validation set
    sorted_files = sorted(zip(train_image_files))
    validation_image_files = random.sample(sorted_files, num_validation)

    # Move corresponding label files to validation directory
    for image_file_tuple in validation_image_files:
        image_file = image_file_tuple[0]  # Extract filename from tuple
        # Construct the paths for the image and label files
        image_path_train = os.path.join(image_folder_train, image_file)
        label_file = os.path.splitext(image_file)[0] + '.txt'
        label_path_train = os.path.join(label_folder_train, label_file)
        image_path_val = os.path.join(image_folder_val, image_file)
        label_path_val = os.path.join(label_folder_val, label_file)
        
        # Move the image and label files to the validation directories
        shutil.move(image_path_train, image_path_val)
        shutil.move(label_path_train, label_path_val)

# exectute function
if __name__ == "__main__":
    image_folder_train = "data/model_training/images/train"
    image_folder_val = "data/model_training/images/val"
    label_folder_train = "data/model_training/labels/train"
    label_folder_val = "data/model_training/labels/val"
    validation_ratio = 0.2

    move_to_validation(image_folder_train, image_folder_val, label_folder_train, label_folder_val, validation_ratio)