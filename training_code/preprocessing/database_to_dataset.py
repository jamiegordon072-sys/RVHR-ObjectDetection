# 22/05/2024
# Script manipulates the database, then extracts the data from a selected database and formats the images and YOLO labels ready for preprocessing
# Creates a folder of images and a folder of labels in the same folder as the database

import sqlite3
import os
import shutil

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



#database_path = "C:/Users/J Gordon/Documents/RVHR/RVHR Full Dataset/4 FB-FOR/240429-025515_M1+M2_T1.db"
#database_path = "C:/Users/J Gordon/Documents/RVHR/RVHR Full Dataset/5 SOT-FB/240429-025115_M1+M2_T1.db"
#database_path = "C:/Users/J Gordon/Documents/RVHR/RVHR Full Dataset/5 UNI-ISB/240430-005054_M1_T1.db"
#database_path = "C:/Users/J Gordon/Documents/RVHR/RVHR Full Dataset/6 ISB-JUNCTION/240430-005855_M1_T1.db"
#database_path = "C:/Users/J Gordon/Documents/RVHR/RVHR Full Dataset/ISB-UNI/240429-035248_M1_T1.db"
#database_path = "C:/Users/J Gordon/Documents/RVHR/RVHR Full Dataset/UNI-KHS/240429-041228_M1_T1.db"
database_path = "C:/Users/J Gordon/Documents/RVHR/RVHR Full Dataset/T1_KGN-MMK_20221114/T1_KGN-MMK_20221114.db"




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
standardise_feature_list_order(database_path, desired_order)

# Convert Spalling Features to Pit
#convert_feature_type(database_path, old_id = 9, new_id = 2)

# Constrain boundary boxes to within image boundary
constrain_boundary_boxes(database_path, image_width = 10000, image_height = 2048)

# Create folder of YOLO annotations
database_to_YOLO(database_path, features_to_omit = ["Squat", "Surface cracks","Crack","Head check","Spalling"], image_width = 10000, image_height = 2048)