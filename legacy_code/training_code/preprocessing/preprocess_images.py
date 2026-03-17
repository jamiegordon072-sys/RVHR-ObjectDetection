# Preprocess images for the training dataset
import os
from pathlib import Path
import sys
import albumentations as A
import numpy as np
from PIL import Image
from PIL import ImageOps
import matplotlib.pyplot as plt
import sqlite3
import os
import shutil

# Dynamically find project root
FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # Move up two levels (training_code/preprocessing)
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # Add ROOT to sys.path
# Import from utils
from utils.preprocessing import open_image, crop_image, split_image, compress_in_x, make_square, transform_annotations


def standardise_feature_list_order(database_path: str, desired_order: list):
    # Changes the order of the feature type list so that feature type ids are uniform between each database

    # Make connection with database
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    # Fetch current data from FeatureType table
    cursor.execute("SELECT id, name FROM FeatureType")
    feature_types = cursor.fetchall()

    # Create a set of current feature type names
    current_names = {name for id, name in feature_types}

    # Insert missing feature types with unique placeholder IDs (-1, -2, -3, etc.)
    next_placeholder_id = -1
    for name in desired_order:
        if name not in current_names:
            cursor.execute("INSERT INTO FeatureType (id, name) VALUES (?, ?)", (next_placeholder_id, name))
            next_placeholder_id -= 1
            conn.commit()  # Commit after each insertion to ensure the changes are saved

    # Fetch updated data from FeatureType table
    cursor.execute("SELECT id, name FROM FeatureType")
    feature_types = cursor.fetchall()

    # Create a dictionary to map current IDs to new IDs based on the desired order
    id_mapping = {}
    current_ids = {name: id for id, name in feature_types}

    for new_id, name in enumerate(desired_order):
        old_id = current_ids.get(name)
        if old_id is not None:
            id_mapping[old_id] = new_id
            
    # Create a temporary mapping system to avoid conflicts
    temp_mapping = {old_id: new_id + 100 for old_id, new_id in id_mapping.items()}

    # Update FeatureType IDs and Feature IDs to temporary IDs
    for old_id, temp_id in temp_mapping.items():
        cursor.execute("UPDATE FeatureType SET id = ? WHERE id = ?", (temp_id, old_id))
        cursor.execute("UPDATE Feature SET ftrType = ? WHERE ftrType = ?", (temp_id, old_id))

    # Adjust FeatureType IDs back to their correct values (subtract 100)
    for temp_id, new_id in id_mapping.items():
        corrected_id = new_id + 100
        cursor.execute("UPDATE FeatureType SET id = ? WHERE id = ?", (new_id, corrected_id))
        cursor.execute("UPDATE Feature SET ftrType = ? WHERE ftrType = ?", (new_id, corrected_id))

    conn.commit()
    conn.close()

def convert_feature_type(database_path:str, old_id: int, new_id: int):
    # Converts all features of one id to another id
    # eg. Convert Spalling (id 8) to Pit (id 3)
    #     Then both Spalling and Pit will have id 3

    # Make connection with database
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    # Make conversion
    cursor.execute("UPDATE Feature SET ftrType = ? WHERE ftrType = ?", (new_id, old_id))

    conn.commit()
    conn.close()

def get_image_dimensions(database_path:str):
    # Get image dimensions from the first image in the database

    # Make connection with database
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    # Fetch image name for image id 1
    cursor.execute("SELECT name FROM Image WHERE id = 1")
    image_name = cursor.fetchone()

    # Search for the image in the left and right rail folders
    base_name = os.path.splitext(os.path.basename(database_path))[0]
    for folder_name in [f"{base_name}_LeftRail", f"{base_name}_RightRail"]:
        image_path = os.path.join(os.path.dirname(database_path), folder_name, image_name[0])
        if os.path.exists(image_path):
            break
    
    # Get image dimensions
    image = open_image(image_path)
    image_shape = image.shape

    return image_shape

def constrain_boundary_boxes(database_path:str, image_width:int, image_height: int):
    # Maximum x and y value for boundary boxes equal are equal to image width and image height
    # Minimum x and y are 0

    # Make connection with database
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    # Fetch current boundary boxes from Feature table
    cursor.execute("SELECT id, x1, y1, x2, y2 FROM Feature")
    features = cursor.fetchall()

    # Constrain boundary boxes to be within the image dimensions
    for feature in features:
        feature_id, x1, y1, x2, y2 = feature
        
        # Constrain coordinates
        x1 = max(0, min(x1, image_width))
        y1 = max(0, min(y1, image_height))
        x2 = max(0, min(x2, image_width))
        y2 = max(0, min(y2, image_height))

        # Ensure x1 < x2 and y1 < y2
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        # Update the database with the constrained values
        cursor.execute("UPDATE Feature SET x1 = ?, y1 = ?, x2 = ?, y2 = ? WHERE id = ?", (x1, y1, x2, y2, feature_id))

    # Commit the changes and close the connection
    conn.commit()
    conn.close()

def database_to_YOLO(database_path: str, features_to_omit: list, image_width: int, image_height: int):
    # Creates a "labels" folder in the database directory containing a text file for every image in the database
    # Each .txt file with have an identical filename to the corresponding .jpg file
    # Each .txt file will contain all associated annotations in YOLO format:
    # feature_id, x_centre, y_centre, width, height
    # given as ratios of the image width rather than pixel number

    # Make connection with database
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    # Fetch the IDs of the feature types to omit
    placeholder = ', '.join(['?'] * len(features_to_omit))
    cursor.execute(f"SELECT id FROM FeatureType WHERE name IN ({placeholder})", features_to_omit)
    omit_feature_ids = [row[0] for row in cursor.fetchall()]

    # Fetch only images with at least one feature and not in omit_feature_ids
    cursor.execute(f"""
        SELECT DISTINCT i.id, i.name
        FROM Image i
        JOIN Feature f ON i.id = f.imageid
        WHERE f.status = 1 AND f.ftrType NOT IN ({','.join('?' for _ in omit_feature_ids)})
    """, omit_feature_ids)
    images = cursor.fetchall()

    # Fetch all active features and their corresponding image IDs
    cursor.execute(f"""
        SELECT imageid, ftrType, x1, y1, x2, y2
        FROM Feature
        WHERE status = 1 AND ftrType NOT IN ({','.join('?' for _ in omit_feature_ids)})
    """, omit_feature_ids)
    features = cursor.fetchall()

    # Ensure image folder exists
    images_folder_path = os.path.join(os.path.dirname(database_path), "images")
    os.makedirs(images_folder_path, exist_ok=True)

    # Ensure labels folder exists
    labels_folder_path = os.path.join(os.path.dirname(database_path), "labels")
    os.makedirs(labels_folder_path, exist_ok=True)

    # Create a dictionary to map image IDs to image names
    image_dict = {image_id: image_name for image_id, image_name in images}

    # Create a dictionary to store annotations for each image
    annotations = {image_id: [] for image_id, _ in images}

    # Process each feature and convert to YOLO format
    for image_id, ftr_type, x1, y1, x2, y2 in features:
        # Constrain coordinates to be within the image dimensions
        x1 = max(0, min(x1, image_width))
        y1 = max(0, min(y1, image_height))
        x2 = max(0, min(x2, image_width))
        y2 = max(0, min(y2, image_height))

        # Calculate YOLO format values
        x_centre = ((x1 + x2) / 2) / image_width
        y_centre = ((y1 + y2) / 2) / image_height
        width = (x2 - x1) / image_width
        height = (y2 - y1) / image_height

        # Append the annotation to the corresponding image's list
        annotations[image_id].append(f"{ftr_type} {x_centre:.6f} {y_centre:.6f} {width:.6f} {height:.6f}")

    # Write annotations to text files and copy images to the "images" folder
    for image_id, ann_list in annotations.items():
        if not ann_list:
            continue  # Skip images with no valid annotations

        image_name = image_dict[image_id]
        txt_filename = os.path.splitext(image_name)[0] + '.txt'
        txt_filepath = os.path.join(labels_folder_path, txt_filename)

        with open(txt_filepath, 'w') as f:
            for ann in ann_list:
                f.write(ann + '\n')

        # Search for the image in the left and right rail folders
        base_name = os.path.splitext(os.path.basename(database_path))[0]
        for folder_name in [f"{base_name}_LeftRail", f"{base_name}_RightRail"]:
            image_path = os.path.join(os.path.dirname(database_path), folder_name, image_name)
            if os.path.exists(image_path):
                # Copy the image to the "images" folder
                shutil.copy(image_path, images_folder_path)
                break

    # Close the database connection
    conn.close()

def extract_images_from_database(database_paths: list):
    # Script manipulates the database, then extracts the data from a selected database and formats the images and YOLO labels ready for preprocessing
    # Creates a folder of images and a folder of labels in the same folder as the database

    # Change order of database to standardised order
    desired_order = [
        "Weld",
        "Corrugation",
        "Microcorrugation",
        "Pit",
        "Block joint",
        "Squat",
        "Surface cracks",
        "Crack",
        "Head check",
        "Spalling"
        ]
    
    # Cycle through runs and extract desired images for each
    for database_path in database_paths:
        #standardise_feature_list_order(database_path, desired_order)

        # Convert Spalling Features to Pit
        #convert_feature_type(database_path, old_id = 9, new_id = 2)

        image_height, image_width = get_image_dimensions(database_path)[:2]

        # Constrain boundary boxes to within image boundary
        constrain_boundary_boxes(database_path, image_width, image_height)

        # Create folder of YOLO annotations
        features_to_omit = ["Squat", "Surface cracks","Crack","Head check","Spalling"]
        database_to_YOLO(database_path, features_to_omit, image_width, image_height)



def load_labels(label_folder_path: str, image_file:str, image: Image):
    annotations = []
    image_name = os.path.splitext(os.path.basename(image_file))[0]
    image_height, image_width = image.shape[:2]
    with open(f"{label_folder_path}/{image_name}.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            # Convert string to list of coordinates
            yolo_bbox = [float(coord) for coord in line.strip().split()]
            # Convert YOLO format to pixel coordinates
            class_id, x_centre, y_centre, bbox_width, bbox_height = yolo_bbox
            x_min = int(image_width*(x_centre - bbox_width/2))
            y_min = int(image_height*(y_centre - bbox_height/2))
            x_max = int(image_width*(x_centre + bbox_width/2))
            y_max = int(image_height*(y_centre + bbox_height/2))
            bbox = [x_min, y_min, x_max, y_max]
            # Make negative values equal to zero
            for j in range(len(bbox)):
                if bbox[j] < 0:
                    bbox[j] = 0
            annotation = (class_id, bbox)
            annotations.append(annotation)
    return annotations

"""
def crop_image(image: Image, annotations: list, num_rakes: int, intensity_threshold_percentage: float):
    # Convert the RGB image to grayscale
    grayscale_image_array = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])

    # Convert the grayscale image array back to uint8 data type
    grayscale_image_array = grayscale_image_array.astype(np.uint8)

    # Get the dimensions of the input image
    image_height, image_width = image.shape[:2]

    lower_thresholds = []
    upper_thresholds = []


    for rake in range(num_rakes):
        # Determine rake position along image
        rake_index = int((rake + 1) * image_width/(num_rakes +1))

        # Extract the pixel intensity line from the image array
        pixel_intensity_line = grayscale_image_array[:, rake_index]

        # Find index where lower threshold is exceeded
        lower_threshold_index = np.argmax(pixel_intensity_line > (np.max(grayscale_image_array) * intensity_threshold_percentage / 100))
        lower_thresholds.append(lower_threshold_index)

        # Find index where upper threshold is exceeded (search from the end of the array)
        upper_threshold_index = np.argmax(pixel_intensity_line[::-1] > (np.max(grayscale_image_array) * intensity_threshold_percentage / 100))
        upper_threshold_index = image_height - upper_threshold_index - 1
        upper_thresholds.append(upper_threshold_index)

    # Find average index where the threshold is first exceeded
    lower_average = sum(lower_thresholds) // len(lower_thresholds)
    upper_average = sum(upper_thresholds) // len(upper_thresholds)

    # Crop image accordingly
    cropped_image = image[lower_average:upper_average, :]

    # Adjust the annotations
    cropped_annotations = []
    for class_id, bbox in annotations:
        x_min, y_min, x_max, y_max = bbox
        # Check if the bounding box intersects with the cropped region
        if y_max > lower_average and y_min < upper_average:
            # Update the coordinates relative to the cropped image
            adjusted_y_min = max(y_min - lower_average, 0)
            adjusted_y_max = min(y_max - lower_average, upper_average - lower_average)
            cropped_annotations.append((class_id, [x_min, adjusted_y_min, x_max, adjusted_y_max]))

    return cropped_image, cropped_annotations

def split_image(image: Image, annotations: list, num_tiles: int, overlap_percentage: float):
     # Get the dimensions of the input image
    image_height, image_width = image.shape[:2]

    # Calculate the width of each tile
    tile_width = image_width // num_tiles

    # Initialize lists to store tiled images and adjusted annotations
    tiled_images = []
    tiled_annotations = []

    # Iterate over each tile
    for i in range(num_tiles):
        # Calculate the left and right coordinates of the current tile
        left = max(int(i * tile_width - overlap_percentage * tile_width / 100), 0)
        right = min(int((i + 1) * tile_width + overlap_percentage * tile_width / 100), image_width)

        # Extract the current tile from the input image
        tile = image[:, left:right]
        tiled_images.append(tile)

        # Adjust the annotations for the current tile
        adjusted_annotations = []
        for class_id, bbox in annotations:
            x_min, y_min, x_max, y_max = bbox
            # Check if the bounding box intersects with the current tile
            if x_min < right and x_max > left:
                # Adjust the coordinates relative to the tile
                adjusted_x_min = max(x_min - left, 0)
                adjusted_y_min = y_min
                adjusted_x_max = min(x_max - left, right - left)
                adjusted_y_max = y_max
                adjusted_annotation = (class_id, [adjusted_x_min, adjusted_y_min, adjusted_x_max, adjusted_y_max])
                adjusted_annotations.append(adjusted_annotation)
        tiled_annotations.append(adjusted_annotations)
    return tiled_images, tiled_annotations

def compress_in_x(image: Image, annotations: list, target_width: int):
    # Get the dimensions of the input image
    image_height, image_width = image.shape[:2]

    # Calculate the ratio of compression
    ratio = target_width / image_width

    target_height = int(image_height * ratio)
    
    # Resize the image
    compressed_image = np.array(Image.fromarray(image).resize((target_width, target_height)))

    
    
    # Adjust bounding box coordinates
    compressed_annotations = []
    for class_id, bbox in annotations:
        x_min, y_min, x_max, y_max = bbox
        # Adjust coordinates according to compression
        adjusted_x_min = int(x_min * ratio)
        adjusted_y_min = int(y_min * ratio)
        adjusted_x_max = int(x_max * ratio)
        adjusted_y_max = int(y_max * ratio)
        compressed_annotations.append((class_id, [adjusted_x_min, adjusted_y_min, adjusted_x_max, adjusted_y_max]))
    
    return compressed_image, compressed_annotations

def  make_square(image: Image, annotations: list, target_height: int):
    # Get the dimensions of the input image
    image_height, image_width = image.shape[:2]
    
    if image_height > target_height:
        # Resize the image
        square_image = np.array(Image.fromarray(image).resize((image_width, target_height)))

        # Calculate the ratio of compression
        ratio = target_height / image_height
        
        # Adjust bounding box coordinates
        square_annotations = []
        for class_id, bbox in annotations:
            x_min, y_min, x_max, y_max = bbox
            # Adjust coordinates according to compression
            adjusted_x_min = x_min  # X-coordinate remains the same
            adjusted_y_min = int(y_min * ratio)
            adjusted_x_max = x_max  # X-coordinate remains the same
            adjusted_y_max = int(y_max * ratio)
            square_annotations.append((class_id, [adjusted_x_min, adjusted_y_min, adjusted_x_max, adjusted_y_max]))
    elif target_height > image_height:
        # Calculate the amount of padding at the top of the image
        pad_top = (target_height - image_height) // 2

        # Pad the image
        square_image = np.array(ImageOps.pad(Image.fromarray(image), (image_width, target_height)))
            
        # Adjust bounding box coordinates
        square_annotations = []
        for class_id, bbox in annotations:
            x_min, y_min, x_max, y_max = bbox
            # Adjust Y-coordinate according to padding
            adjusted_y_min = int(y_min + pad_top)
            adjusted_y_max = int(y_max + pad_top)
            square_annotations.append((class_id, [x_min, adjusted_y_min, x_max, adjusted_y_max]))

    return square_image, square_annotations
"""
def augment_images(image: Image, annotations:list, num_augmentations: int):
    # Get the dimensions of the input image
    image_height, image_width = image.shape[:2]

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
    ], bbox_params=A.BboxParams(format='albumentations', min_visibility=0.3, label_fields=['category_id']))

    # Generate multiple augmented images
    augmented_images = []
    augmented_annotations = []
    for i in range(num_augmentations):
    
        bboxes = []
        class_ids = []
        for class_id, bbox in annotations:
            # Convert annotations to Albumentations format (x_center, y_center, width, height)
            bbox = [(bbox[0] / image_width), (bbox[1] / image_height), (bbox[2] / image_width), (bbox[3] / image_height)]
            
            # Check bbox dimensions are valid 
            if bbox[2]>0 and bbox[3]>0:
                class_ids.append(class_id)
                bboxes.append(bbox)
            
        

        # Apply augmentation to the image and annotations
        transformed = transform(image=image, bboxes=bboxes, category_id=[1]*len(bboxes))

        # Convert the augmented annotations back to pixel format
        adjusted_annotations = []
        for class_id, bbox in zip(class_ids, transformed["bboxes"]):
            augmented_bbox = [int(bbox[0] * image_width), int(bbox[1] * image_height), int(bbox[2] * image_width), int(bbox[3] * image_height)]
            adjusted_annotations.append((class_id, augmented_bbox))
        augmented_images.append(transformed["image"])
        augmented_annotations.append(adjusted_annotations)
    return augmented_images, augmented_annotations


    #Convert pixel annotations to YOLO format

    # Save YOLO annotations to augmented label folder path

def save_labels(image: Image, annotations: list, annotation_file: str):
    # Get the dimensions of the input image
    image_height, image_width = image.shape[:2]

    # Convert pixel annotations to YOLO format
    yolo_annotations = []
    for class_id, bbox in annotations:
        x_min, y_min, x_max, y_max = bbox
        x_centre = (x_min + x_max) / (2 * image_width)
        y_centre = (y_min + y_max) / (2 * image_height)
        bbox_width = (x_max - x_min) / image_width
        bbox_height = (y_max - y_min) / image_height
        yolo_annotation = [class_id, x_centre, y_centre, bbox_width, bbox_height]
        yolo_annotations.append(yolo_annotation)

    # Save YOLO annotations
    with open(annotation_file, "w") as f:
        for yolo_annotation in yolo_annotations:
            # Convert YOLO format to string and write to file
            line = " ".join(str(coord) for coord in yolo_annotation)
            f.write(line + "\n")

def create_training_dataset(database_paths: list):
    
    # Cycle through each run and preprocess selected images
    for database_path in database_paths:
        image_folder_path = os.path.join(os.path.dirname(database_path), 'images')
        label_folder_path = os.path.join(os.path.dirname(database_path), 'labels')
        new_image_folder_path = "data/model_training/images/train"
        new_label_folder_path = "data/model_training/labels/train"


        # Define parameters
        num_tiles = 5
        image_size = [640, 640]
        num_augmentations = 2

        # Create list of image files in image folder
        image_paths = [os.path.join(image_folder_path, file) for file in os.listdir(image_folder_path) if file.endswith((".jpg", ".jpeg", ".png"))]

        for image_path in image_paths:
            # Get image and convert to a NumPy array
            image = open_image(image_path)
            
            # Load corresponding YOLO labels and convert to pixel format
            original_annotations = load_labels(label_folder_path, image_path, image)

            # Crop image to width of rail
            image, (crop_y_min, crop_y_max) = crop_image(image)
            
            # Split images into tiles
            image_tiles, tile_boundaries = split_image(image, num_tiles)

            for tile_index, (image_tile, tile_boundary) in enumerate(zip(image_tiles, tile_boundaries)):
                # Compress images to a specified width
                image_tile, x_compression = compress_in_x(image_tile, image_size[1])
            
                # Make images square
                image_tile, y_compression, y_padding = make_square(image_tile, image_size[0])
                
                # Transform annotations to match processed image
                transformation_map = ((crop_y_min, crop_y_max), tile_boundary, x_compression, y_compression, y_padding)
                adjusted_annotations = transform_annotations(original_annotations, transformation_map)
            
                # Augment images
                augmented_images, augmented_annotations = augment_images(image_tile, adjusted_annotations, num_augmentations)
                #augmented_images, augmented_annotations = [image_tile], [adjusted_annotations]

                for augmented_index, (augmented_image, augmented_annotation) in enumerate(zip(augmented_images, augmented_annotations)):
                    """"""
                    plt.figure()  # Create a new figure for each tile
                    plt.imshow(augmented_image)
                    plt.axis('off')  # Hide axis
                    # Overlay bounding boxes
                    for class_id, bbox in augmented_annotation:
                        x_min, y_min, x_max, y_max = bbox
                        width = x_max - x_min
                        height = y_max - y_min
                        rect = plt.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='r', facecolor='none')
                        plt.gca().add_patch(rect)
                    plt.show()
                    

                    # Generate name for the current image
                    image_name = f"{os.path.splitext(os.path.basename(image_path))[0]}_tile{tile_index}_augmented{augmented_index}"
                    
                    # Convert numpy array to PIL Image
                    augmented_image_pil = Image.fromarray(augmented_image)
                    
                    # Save image to path
                    #augmented_image_pil.save(f"{new_image_folder_path}/{image_name}.jpg")
                    
                    # Save labels to path
                    #save_labels(augmented_image, augmented_annotation, f"{new_label_folder_path}/{image_name}.txt")

def crop_split_dataset():
    # Define paths to images and labels
    image_folder_path = "data/original/T1-KGN/images"
    label_folder_path = "data/original/T1-KGN/labels"
    new_image_folder_path = "data/model_testing/split_test"
    new_label_folder_path = "data/corrugation/T1-KGN/split_test"


    # Define parameters
    num_rakes = 10
    intesity_threshold_percentage = 20
    num_tiles = 5
    overlap_percentage = 20

    # Create list of image files in image folder
    image_files = [file for file in os.listdir(image_folder_path) if file.endswith((".jpg", ".jpeg", ".png"))]

    for image_file in image_files:
        # Get image and convert to a NumPy array
        image = np.array(Image.open(os.path.join(image_folder_path, image_file)))
        
        # Load corresponding YOLO labels and convert to pixel format
        original_annotations = load_labels(label_folder_path, image_file, image)

        # Crop image to width of rail
        cropped_image, cropped_annotations = crop_image(image, original_annotations, num_rakes, intesity_threshold_percentage)
        
        # Split images into tiles
        tiled_images, tiled_annotations = split_image(cropped_image, cropped_annotations, num_tiles, overlap_percentage)

        for tile_index, (tiled_image, tiled_annotation) in enumerate(zip(tiled_images, tiled_annotations)):
            # Generate name for the current image
            image_name = f"{os.path.splitext(os.path.basename(image_file))[0]}_tile{tile_index}"
                
            # Convert numpy array to PIL Image
            image_pil = Image.fromarray(tiled_image)
                
            # Save image to path
            image_pil.save(f"{new_image_folder_path}/{image_name}.jpg")
                
            # Save labels to path
            # save_labels(tiled_image, tiled_annotation, f"{new_label_folder_path}/{image_name}.txt")

def create_test_dataset():
    # Define paths to images and labels
    image_folder_path = "data/model_testing/general_test"
    label_folder_path = "data/model_testing/general_test"
    new_image_folder_path = "data/model_testing/new_data_test"
    new_label_folder_path = "data/unspecified_path"


    # Define parameters
    num_rakes = 10
    intesity_threshold_percentage = 20
    num_tiles = 5
    overlap_percentage = 20
    image_size = [640, 640]


    # Create list of image files in image folder
    image_files = [file for file in os.listdir(image_folder_path) if file.endswith((".jpg", ".jpeg", ".png"))]

    for image_file in image_files[:36]:
        # Get image and convert to a NumPy array
        image = np.array(Image.open(os.path.join(image_folder_path, image_file)))
        
        # Load corresponding YOLO labels and convert to pixel format
        original_annotations = load_labels(label_folder_path, image_file, image)

        # Crop image to width of rail
        cropped_image, cropped_annotations = crop_image(image, original_annotations, num_rakes, intesity_threshold_percentage)
        
        # Split images into tiles
        tiled_images, tiled_annotations = split_image(cropped_image, cropped_annotations, num_tiles, overlap_percentage)

        for tile_index, (tiled_image, tiled_annotation) in enumerate(zip(tiled_images, tiled_annotations)):
            # Compress images to a specified width
            compressed_image, compressed_annotations = compress_in_x(tiled_image, tiled_annotation, image_size[1])
            
            # Make images square
            square_image, square_annotations = make_square(compressed_image, compressed_annotations, image_size[0])

            # Generate name for the current image
            image_name = f"{os.path.splitext(os.path.basename(image_file))[0]}_tile{tile_index}"
                
            # Convert numpy array to PIL Image
            image_pil = Image.fromarray(square_image)
                
            # Save image to path
            image_pil.save(f"{new_image_folder_path}/{image_name}.jpg")
                
            # Save labels to path
            # save_labels(tiled_image, tiled_annotation, f"{new_label_folder_path}/{image_name}.txt")

# Define paths to images and labels
#database_paths = ["C:/Users/J Gordon/Documents/RVHR/RVHR Full Dataset/4 FB-FOR/240429-025515_M1+M2_T1.db",
#             "C:/Users/J Gordon/Documents/RVHR/RVHR Full Dataset/5 SOT-FB/240429-025115_M1+M2_T1.db",
#             "C:/Users/J Gordon/Documents/RVHR/RVHR Full Dataset/5 UNI-ISB/240430-005054_M1_T1.db",
#             "C:/Users/J Gordon/Documents/RVHR/RVHR Full Dataset/6 ISB-JUNCTION/240430-005855_M1_T1.db",
#             "C:/Users/J Gordon/Documents/RVHR/RVHR Full Dataset/ISB-UNI/240429-035248_M1_T1.db",
#             "C:/Users/J Gordon/Documents/RVHR/RVHR Full Dataset/UNI-KHS/240429-041228_M1_T1.db",
#             "C:/Users/J Gordon/Documents/RVHR/RVHR Full Dataset/T1_KGN-MMK_20221114/T1_KGN-MMK_20221114.db"]
database_paths = ["C:/Users/J Gordon/Documents/Rail Tech/RV-HR/Data/13 Rhp - Kh - Copy - Copy/241010-020838_RHP-KH_2_021.db"]

extract_images_from_database(database_paths)

create_training_dataset(database_paths)
#crop_split_dataset()
#create_test_dataset()