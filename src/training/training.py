import os
import cv2
import albumentations as A
import hashlib
import yaml

from models.detection import Detection, ImageData
from models.transformation import TransformationMap


def annotations_to_tile(image_tile: ImageData, trans_map: TransformationMap, annotations: list[Detection]) -> ImageData:
    """
    Adjust bounding box according to the transformation map to realign with the preprocessed image

    :param annoation: Original Bounding box annotation in YOLO format
    :param transformation_map: List of the relevant metrics used in the preprocessing of the image.
    :return: adjusted_annotation: ImageTile object containing the adjusted annotations
    """


    # Step 1: Crop in Y
    y_crop_annotations = []
    for annotation in annotations:
        x_min, y_min, x_max, y_max = annotation.bbox

        if y_max > trans_map.crop_y_min and y_min < trans_map.crop_y_max:
            new_y_min = max(
                y_min - trans_map.crop_y_min,
                0
            )
            new_y_max = min(
                y_max - trans_map.crop_y_min,
                trans_map.crop_y_max - trans_map.crop_y_min
            )

            y_crop_annotations.append(
                Detection(
                    label=annotation.label,
                    bbox=[x_min, new_y_min, x_max, new_y_max]
                )
            )

    # Step 2: Tile filter
    tile_annotations = []
    for annotation in y_crop_annotations:
        x_min, y_min, x_max, y_max = annotation.bbox

        if x_min < trans_map.tile_x_max and x_max > trans_map.tile_x_min:
            new_x_min = max(
                x_min - trans_map.tile_x_min,
                0
            )
            new_x_max = min(
                x_max - trans_map.tile_x_min,
                trans_map.tile_x_max - trans_map.tile_x_min
            )

            tile_annotations.append(
                Detection(
                    label=annotation.label,
                    bbox=[new_x_min, y_min, new_x_max, y_max]
                )
            )

    # Step 3: Scale + pad
    processed_annotations = []
    for annotation in tile_annotations:
        x_min, y_min, x_max, y_max = annotation.bbox

        new_x_min = x_min * trans_map.x_compression
        new_y_min = y_min * trans_map.x_compression * trans_map.y_compression + trans_map.y_padding
        new_x_max = x_max * trans_map.x_compression
        new_y_max = y_max * trans_map.x_compression * trans_map.y_compression + trans_map.y_padding

        processed_annotations.append(
            Detection(
                label=annotation.label,
                bbox=[new_x_min, new_y_min, new_x_max, new_y_max]
            )
        )

    # Update the image tile with the selected processed annotations
    image_tile.detections = processed_annotations

    return image_tile


def augment_images(image_tile: ImageData, num_augmentations: int) -> list[ImageData]:
    """
    Apply data augmentation techniques to the input image and adjust annotations accordingly.
    
    :param image: Input image as a PIL Image object.
    :param annotations: List of Detection objects containing bounding box annotations.
    :param num_augmentations: Number of augmented images to generate.
    :return: List of augmented images with their corresponding adjusted annotations
    """

    # Define a transformation pipeline
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),  # horizontally flip with probability 0.5
        A.GaussianBlur(p=0.5),     # apply Gaussian blur with probability 0.5
        A.Rotate(p=0.5, border_mode=0),  # rotate image between -90 and 90 degrees with probability 0.5, border mode blacks out surroundings
        A.RandomBrightnessContrast(p=0.5),  # randomly change brightness and contrast
        A.RandomGamma(p=0.5),       # randomly change gamma
        A.RandomBrightnessContrast(p=0.5),  # randomly change brightness and contrast
        A.RGBShift(p=0.5),          # randomly shift RGB channels
        A.Blur(p=0.5),              # blur image with probability 0.5
        ],
        bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.3, label_fields=['category_id'])
        )

    # Generate multiple augmented images
    augmented_images = []
    for aug_idx in range(num_augmentations):
    
        # Get list of bboxes
        bboxes = [annotation.bbox for annotation in image_tile.detections]

        # Apply augmentation to the image and annotations
        transformed = transform(image=image_tile.image, bboxes=bboxes, category_id=[1]*len(bboxes))

        # Get list of adjusted annotations
        augmented_annotations =[]
        for bbox_idx, bbox in enumerate(transformed['bboxes']):
            augmented_annotation = Detection(label=image_tile.detections[bbox_idx].label,
                                               bbox=bbox
                                            )
            augmented_annotations.append(augmented_annotation)
        
        # Create new image object with new image and adjusted annotations
        augmented_image = ImageData(
            image_tag=f"{image_tile.image_tag}_aug{aug_idx}",
            image=transformed["image"],
            detections=augmented_annotations
        )
        augmented_images.append(augmented_image)

    return augmented_images


def get_yolo_bbox(annotation: Detection, image_width: int, image_height: int):
    """
    Convert bounding box annotation to YOLO format (x_center, y_center, width, height) normalized by image dimensions.
    
    :param annotation: Detection object containing bounding box annotation.
    :param image_width: Width of the image in pixels.
    :param image_height: Height of the image in pixels.
    :return: Tuple containing (x_center, y_center, width, height) in YOLO format.
    """
    
    x_min, y_min, x_max, y_max = annotation.bbox
    
    x_center = (x_min + x_max) / 2 / image_width
    y_center = (y_min + y_max) / 2 / image_height
    width = (x_max - x_min) / image_width
    height = (y_max - y_min) / image_height
    
    return x_center, y_center, width, height


def save_annotations(annotations: list[Detection], save_path: str, image_width: int, image_height: int):
    """
    Save the annotation to the specified output path.
    
    :param annotation: Detection object containing bounding box annotation.
    :param output_path: Path to save the annotation.
    :return: None
    """
    
    # Save the annotation to a text file in YOLO format
    with open(save_path, 'w') as f:
        for annotation in annotations:
            ftr_type = annotation.label
            x_centre, y_centre, width, height = get_yolo_bbox(annotation, image_width, image_height)
            f.write(f"{ftr_type} {x_centre:.6f} {y_centre:.6f} {width:.6f} {height:.6f}\n")


def is_val(image_tag: str, val_ratio: float = 0.2) -> bool:
    """
    Determine whether image goes into train or val set
    Not random so that tiles from the same image will always be either train or val. Prevents Data Leaking between train and val
    """
    base = image_tag.split("_tile")[0]
    h = int(hashlib.md5(base.encode()).hexdigest(), 16)
    return (h % 100) < int(val_ratio * 100)


def load_split_file(filepath: str) -> set[str]:
    """
    Load a YOLO split file into a set of image paths. Returns empty set if file does not exist.
    
    param filepath: Path to train.txt or val.txt
    returns: Set of image paths in YOLO split file
    """
    if not os.path.exists(filepath):
        return set()

    with open(filepath, "r") as f:
        return set(line.strip() for line in f if line.strip())


def save_training_data(image_objects: list[ImageData], dataset_folder: str):
    """
    Save the augmented images and annotations to the specified output path.
    
    :param images: List of augmented images as PIL Image objects.
    :param annotations: List of lists of Detection objects containing bounding box annotations for each image.
    :param output_path: Path to save the augmented images and annotations.
    :return: None
    """

    # Define Master Dataset
    img_folder = os.path.join(dataset_folder, "all", "images")
    lbl_folder = os.path.join(dataset_folder, "all", "labels")

    # Define 
    split_folder = os.path.join(dataset_folder, "splits")
    train_file = os.path.join(split_folder, "train.txt")
    val_file = os.path.join(split_folder, "val.txt")

    # Create all directories
    if os.path.isdir(dataset_folder):
        for folder in [img_folder, lbl_folder, split_folder]:
            os.makedirs(folder, exist_ok=True)
    else:
        raise ValueError(f"Dataset Folder does not exist at {dataset_folder}")


    train_lines = []
    val_lines = []
    for image_object in image_objects:
        image = image_object.image
        tag = image_object.image_tag

        image_path = os.path.join(img_folder, f"{tag}.jpg")
        label_path = os.path.join(lbl_folder, f"{tag}.txt")

        # Skip if already exists (prevents duplicates)
        if os.path.exists(image_path):
            continue

        # Save to master dataset
        cv2.imwrite(image_path, image)

        h, w = image.shape[0], image.shape[1]
        save_annotations(image_object.detections, label_path, w, h)

        # Assign split
        if is_val(tag):
            val_lines.append(image_path)
        else:
            train_lines.append(image_path)

    # Remove any new entries already in splits .txts
    existing_train = load_split_file(train_file)
    existing_val = load_split_file(val_file)
    all_existing = existing_train.union(existing_val)
    train_lines = [l for l in train_lines if l not in all_existing]
    val_lines = [l for l in val_lines if l not in all_existing]

    # Append new entries
    with open(train_file, "a") as f:
        for line in train_lines:
            f.write(line + "\n")

    with open(val_file, "a") as f:
        for line in val_lines:
            f.write(line + "\n")


def generate_data_yaml(dataset_folder: str, labels: dict[int, str]):
    """
    Generate YOLO data.yaml file from dataset folder and label dict
    """

    yaml_path = os.path.join(dataset_folder, "data.yaml")

    data = {
        "path": os.path.abspath(dataset_folder),
        "train": "splits/train.txt",
        "val": "splits/val.txt",
        "names": labels
    }

    with open(yaml_path, "w") as f:
        yaml.dump(data, f, sort_keys=False)
