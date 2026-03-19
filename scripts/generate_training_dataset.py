import os
from tqdm import tqdm

from database.Database import RVHR_DB
from preprocessing.preprocess import preprocess_image
from training.training import annotations_to_tile, augment_images, save_training_data, generate_data_yaml
from utils.utils import find_image_path



# Define Output Training Dataset Path
TRAINING_DATASET_PATH = "data/training_dataset"

# Define labels
LABELS = {
    0: "Weld",
    1: "Corrugation",
    2: "Pit",
    3: "Block Joint"
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
    img_ids_1 = db.get_img_ids_manual_labels(ftr_types) # Get Images with manual labels
    img_ids_2 = db.get_img_ids_deleted_labels(ftr_types) # Get Images with automatic labels removed
    img_ids = list(set(img_ids_1) | set(img_ids_2))
    
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

    # Create a YOLO data.yaml file in the training dataset folder
    generate_data_yaml(TRAINING_DATASET_PATH, LABELS)

if __name__ == "__main__":

    db_paths = [
        "C:/Users/J Gordon/Documents/RVHR/RVHR Full Dataset/4 FB-FOR/240429-025515_M1+M2_T1.db",
        "C:/Users/J Gordon/Documents/RVHR/RVHR Full Dataset/5 SOT-FB/240429-025115_M1+M2_T1.db",
        "C:/Users/J Gordon/Documents/RVHR/RVHR Full Dataset/5 UNI-ISB/240430-005054_M1_T1.db",
        "C:/Users/J Gordon/Documents/RVHR/RVHR Full Dataset/6 ISB-JUNCTION/240430-005855_M1_T1.db",
        "C:/Users/J Gordon/Documents/RVHR/RVHR Full Dataset/ISB-UNI/240429-035248_M1_T1.db",
        "C:/Users/J Gordon/Documents/RVHR/RVHR Full Dataset/UNI-KHS/240429-041228_M1_T1.db",
        "C:/Users/J Gordon/Documents/RVHR/RVHR Full Dataset/T1_KGN-MMK_20221114/T1_KGN-MMK_20221114.db"]
    
    db_paths = ["C:/Workspace/Rail Tech/RV-HR/08 - Sample Data/Data/CPH 032026/260303-025143_M3_T2.db"]
    for db_path in db_paths:
        generate_training_dataset(db_path)