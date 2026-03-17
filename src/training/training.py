import numpy as np
import cv2
import albumentations as A

from src.models.detection import Detection, ImageData
from src.models.transformation import TransformationMap


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
            f.write(f"{ftr_type} {x_centre:.6f} {y_centre:.6f} {width:.6f} {height:.6f}")


def save_training_data(image_objects: list[ImageData], output_folder: str):
    """
    Save the augmented images and annotations to the specified output path.
    
    :param images: List of augmented images as PIL Image objects.
    :param annotations: List of lists of Detection objects containing bounding box annotations for each image.
    :param output_path: Path to save the augmented images and annotations.
    :return: None
    """

    for image_object in image_objects:
        # Save the augmented image
        image = image_object.image
        image_path = f"{output_folder}/{image_object.image_tag}.jpg"
        cv2.imwrite(image_path, image)

        # Save the corresponding annotation
        image_width, image_height = image.shape[1], image.shape[0]
        annotation_path = f"{output_folder}/{image_object.image_tag}.txt"
        save_annotations(image_object.detections, annotation_path, image_width, image_height)

