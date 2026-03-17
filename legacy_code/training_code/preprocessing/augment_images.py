import albumentations as A
from PIL import Image
import numpy as np
import os

#define function to load YOLO annotations from a .txt file
def load_yolo_annotations(annotation_file):
    annotations = []
    with open(annotation_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            # Convert string to list of coordinates
            bbox = [float(coord) for coord in line.strip().split()]
            annotations.append(bbox)
    return annotations

#define function to save YOLO annotations to a .txt file
def save_yolo_annotations(annotations, annotation_file):
    with open(annotation_file, "w") as f:
        for bbox in annotations:
            # Convert YOLO format to string and write to file
            line = " ".join(str(coord) for coord in bbox)
            f.write(line + "\n")

# Define the number of augmented images you want to generate for each original image
num_augmented_images = 3


#Path to folder containing images
image_folder_path = "data/original/T1-PHP/images"
annotation_folder_path = "data/original/T1-PHP/labels"

# Get the list of files in the folder
image_files = [file for file in os.listdir(image_folder_path) if file.endswith((".jpg", ".jpeg", ".png"))]

for image_file in image_files:
    # Load your image
    image_name = os.path.splitext(os.path.basename(image_file))[0]
    image = np.array(Image.open(f"{image_folder_path}/{image_name}.jpeg"))
    annotations = load_yolo_annotations(f"{annotation_folder_path}/{image_name}.txt")

    # Define a transformation pipeline
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),  # horizontally flip with probability 0.5
        A.GaussianBlur(p=0.5),     # apply Gaussian blur with probability 0.5
        #A.Rotate(p=0.5, border_mode=0),  # rotate image between -90 and 90 degrees with probability 0.5, border mode blacks out surroundings
        A.RandomBrightnessContrast(p=0.5),  # randomly change brightness and contrast
        A.RandomGamma(p=0.5),       # randomly change gamma
        A.RandomBrightnessContrast(p=0.5),  # randomly change brightness and contrast
        A.RGBShift(p=0.5),          # randomly shift RGB channels
        A.Blur(p=0.5),              # blur image with probability 0.5
    ], bbox_params=A.BboxParams(format='albumentations', min_visibility=0.3, label_fields=['category_id']))

    # Generate and save multiple augmented images
    for i in range(num_augmented_images):
    
        bboxes = []
        label_classes = []
        for bbox in annotations:
            label_class = bbox[0]
            # Convert YOLO annotations to Albumentations format (x_center, y_center, width, height)
            bbox = [(bbox[1] - bbox[3] / 2), (bbox[2] - bbox[4] / 2), (bbox[1] + bbox[3] / 2), (bbox[2] + bbox[4] / 2)]
            for j in range(len(bbox)):
                if bbox[j] < 0:
                    bbox[j] = 0
            bboxes.append(bbox)
            label_classes.append(label_class)

        # Apply augmentation to the image and annotations
        transformed = transform(image=image, bboxes=bboxes, category_id=[1]*len(bboxes))

        # Convert the augmented annotations back to YOLO format
        index = 0
        augmented_bboxes = []
        for bbox in transformed["bboxes"]:
            augmented_bbox = [label_classes[index], (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2, bbox[2] - bbox[0], bbox[3] - bbox[1]]
            augmented_bboxes.append(augmented_bbox)
            index +=1

        # Get the augmented image
        augmented_image = transformed["image"]
        
        # Convert numpy array to PIL Image
        augmented_image_pil = Image.fromarray(augmented_image)
        
        # Save the augmented image with a unique filename
        augmented_image_pil.save(f"{image_folder_path}/{image_name}_augmented_{i}.jpg")

        # Save the augmented image and annotations
        save_yolo_annotations(augmented_bboxes, f"{annotation_folder_path}/{image_name}_augmented_{i}.txt")
