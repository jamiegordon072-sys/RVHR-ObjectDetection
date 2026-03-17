import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

from src.models.detection import ImageData
from src.utils.utils import get_image, load_annotations

def display_features(image_object: ImageData):
    """
    Display the features on the image for visualization purposes.
    """

    image_tag = image_object.image_tag
    image = image_object.image
    annotations = image_object.detections

    for annotation in annotations:
        x_min, y_min, x_max, y_max = annotation.bbox
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 4) # Draw bounding box
        # Optionally, add label text
        cv2.putText(image, str(annotation.label), (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 4)
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.imshow(image_rgb)
    plt.title(image_tag)
    plt.axis("off")
    plt.show()

if __name__ == "__main__":

    image_path = "training_dataset/M3_T2_021-00009_20260303_00007_200,670km_xmin4198_xmax7462_aug0.jpg"
    annotations_path = "training_dataset/M3_T2_021-00009_20260303_00007_200,670km_xmin4198_xmax7462_aug0.txt"

    image = get_image(image_path)
    annotations = load_annotations(annotations_path, image.shape[1], image.shape[0])
    image_object = ImageData(
            image_tag=os.path.splitext(os.path.basename(image_path))[0],
            image=image,
            detections=annotations
        )
    display_features(image_object)