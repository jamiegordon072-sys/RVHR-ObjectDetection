import cv2
import os
import matplotlib.pyplot as plt

from models.detection import ImageData
from utils.utils import get_image, load_annotations

def display_features(image_object: ImageData):
    """
    Display the features on the image for visualization purposes.
    """

    image_tag = image_object.image_tag
    image = image_object.image
    
    for detection in image_object.detections:
        x_min, y_min, x_max, y_max = detection.bbox
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 4) # Draw bounding box
        # Optionally, add label text
        cv2.putText(image, str(detection.label), (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 4)
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.imshow(image_rgb)
    plt.title(image_tag)
    plt.axis("off")
    plt.show()

if __name__ == "__main__":

    image_path = "data/training_dataset/all/images/M1_T1_LeftRail_20240429_00040_1,933km_tile_1600_4400_aug1.jpg"
    labels_path = "data/training_dataset/all/labels/M1_T1_LeftRail_20240429_00040_1,933km_tile_1600_4400_aug1.txt"

    image = get_image(image_path)
    annotations = load_annotations(labels_path, image.shape[1], image.shape[0])
    image_object = ImageData(
            image_tag=os.path.splitext(os.path.basename(image_path))[0],
            image=image,
            detections=annotations
        )
    display_features(image_object)