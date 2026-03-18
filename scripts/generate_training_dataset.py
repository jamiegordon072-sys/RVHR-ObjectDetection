import os
from tqdm import tqdm

from src.database.Database import RVHR_DB
from src.preprocessing.preprocess import preprocess_image
from src.training.training import annotations_to_tile, augment_images, save_training_data
from src.utils.utils import find_image_path
from src.models.detection import ImageData



# Define Output Training Dataset Path
TRAINING_DATASET_PATH = "training_dataset"

# Define labels
LABELS = {
    0: "Weld",
    1: "Corrugation",
    2: "Pit",
    3: "Block Joint"
}
""""""
LABELS = {
    0: "Pit",
    1: "Weld",
    2: "F2"
}

def generate_training_dataset(db_path:str):
    """
    Generate training dataset by cropping and splitting images, and applying data augmentation.
    
    :param db_path: Path to the database containing image information and annotations.
    :return: None
    """

    # Connect to the DB
    db = RVHR_DB(db_path)

    source_dir = os.path.dirname(db_path)

    # Standardize Feature List Table to match labels
    db.reorder_feature_types(LABELS)

    # Get list of all images with annotations
    ftr_types = list(LABELS.keys())
    img_ids = db.get_labelled_img_ids(ftr_types)
    
    for img_id in tqdm(img_ids, desc="Processing Images", unit="img"):
        # Get image path
        img_name = db.get_img_name(img_id)
        img_path = find_image_path(img_name, source_dir)
        if img_path is None:
            print(f"Image {img_name} not found in source directory. Skipping.")
            continue
        
        # Get annotations
        annotations = db.get_labelled_features(img_id)

        # Preprocess the image and get transformation map
        image_tiles, trans_maps = preprocess_image(img_path)

        training_images =[]
        for image_tile, trans_map in zip(image_tiles, trans_maps):
            image_tile = annotations_to_tile(image_tile, trans_map, annotations)

            # Augment the preprocessed image and annotations
            augmented_image_tiles = augment_images(image_tile, num_augmentations=2)

            training_images.extend(augmented_image_tiles)

        # Save the training images and annotations to the training dataset path
        save_training_data(training_images, TRAINING_DATASET_PATH)

if __name__ == "__main__":
    db_path = "C:/Workspace/Rail Tech/RV-HR/Data/CPH 032026/260303-025143_M3_T2.db"
    generate_training_dataset(db_path)