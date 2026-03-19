import os
import cv2

from models.detection import Detection

def get_image(image_path):
    # Load image as BGR format using OpenCV
    return cv2.imread(image_path, cv2.IMREAD_COLOR)

def get_image_tag(image_path):
    # Get the basename of the image without the extension
    return os.path.splitext(os.path.basename(image_path))[0]


def find_image_path(filename: str, source_dir: str) -> str | None:
    """
    Search for an image file inside source_dir and its subdirectories.

    param filename: Name of the image file (e.g. "image1.jpg")
    param source_dir: Directory to search
    return: Full path to the file if found, otherwise None
    """

    for root, _, files in os.walk(source_dir):
        if filename in files:
            return os.path.join(root, filename)
    return None


def load_annotations(annotation_path: str, image_width: int, image_height: int) -> list[Detection]:
    """
    Load annotations from a YOLO format text file
    param annotation_path: Path to the annotation text file
    return: List of Detection objects
    """

    annotations = []
    with open(annotation_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue  # Skip invalid lines
            ftr_type = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            # Convert from YOLO format to pixel coordinates
            x_min = int((x_center - width / 2) * image_width)
            y_min = int((y_center - height / 2) * image_height)
            x_max = int((x_center + width / 2) * image_width)
            y_max = int((y_center + height / 2) * image_height)

            annotation = Detection(
                label=ftr_type,
                bbox=(x_min, y_min, x_max, y_max)
                )

            annotations.append(annotation)
    return annotations

