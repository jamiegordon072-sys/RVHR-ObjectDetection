# -*- coding: utf-8 -*-

# analysis_server.py

# Imports
import socket
import sys
import logging
import os
import numpy as np
from datetime import datetime
from ultralytics import YOLO
import sqlite3
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

# Configure logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("executable_progress_log.log"),
                        logging.StreamHandler(sys.stdout)
                    ])
        
def crop_image(image: Image, num_rakes: int, intensity_threshold_percentage: float):

    """
    Crops image down to width of the rail
    Rail is detected by taking vertical rakes down the image at regular intervals and analysing the change in intensity.
    Where the threshold is exceeded the rail is detected
    """

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

    return cropped_image, lower_average

def split_image(image: Image, num_tiles: int, overlap_percentage: float):

    """
    Image is split into tiles of equal size.
    There is an overlap between adjacent tiles.
    The first and last tile will be smaller since they have overlap on only one side.
    """

     # Get the dimensions of the input image
    image_height, image_width = image.shape[:2]

    # Calculate the width of each tile
    tile_width = image_width // num_tiles

    # Initialize lists to store tiled images and adjusted annotations
    tiled_images = []
    tile_offsets = []

    # Iterate over each tile
    for i in range(num_tiles):
        # Calculate the left and right coordinates of the current tile
        left = max(int(i * tile_width - overlap_percentage * tile_width / 100), 0)
        right = min(int((i + 1) * tile_width + overlap_percentage * tile_width / 100), image_width)

        # Extract the current tile from the input image
        tile = image[:, left:right]
        tiled_images.append(tile)
        tile_offsets.append(left)

    return tiled_images, tile_offsets

def compress_in_x(image: Image, target_width: int):

    """
    Image is compressed to a target width.
    Aspect ratio is maintained so image is compressed in both x and y.
    """

    # Get the dimensions of the input image
    image_height, image_width = image.shape[:2]

    # Calculate the ratio of compression
    compression_ratio = target_width / image_width

    target_height = int(image_height * compression_ratio)
    
    # Resize the image
    compressed_image = np.array(Image.fromarray(image).resize((target_width, target_height)))
    
    return compressed_image, compression_ratio

def  make_square(image: Image, target_height: int):

    """
    Image is made square [assuming target_height is equal to the image width].
    If image height is less than target height then the image is padded equally on top and bottom to make target height.
    If image height is greater than target height then the image is compressed in y. Aspect ratio is not maintained
    """

    # Get the dimensions of the input image
    image_height, image_width = image.shape[:2]
    
    if image_height > target_height:
        y_compression = target_height/image_height
        y_padding = 0
        # Resize the image
        square_image = np.array(Image.fromarray(image).resize((image_width, target_height)))
    elif target_height > image_height:
        # Calculate the amount of padding at the top of the image
        y_padding = (target_height - image_height) // 2
        y_compression = 1
        # Pad the image
        square_image = np.array(ImageOps.pad(Image.fromarray(image), (image_width, target_height)))

    return square_image, y_compression, y_padding

def format_results(results: list, x_compression: float, y_compression: float, y_padding: int, vertical_offset: int, tile_offset: int):

    """
    Results are put in correct format.
    The results from the prepocessed tiled images are converted into the coordinate system of the original full size image.
    The results for the different tiles are compiled into a single variable.
    Each result are put in the format [Feature Type, Xmin, Ymin, Xmax, Ymax, Confidence]
    """

    predicted_boxes = []
    for result in results:
        # Extract bounding boxes from the 'boxes' attribute
        boxes = result.boxes.xyxy.tolist()  # [xmin, ymin, xmax, ymax]
        if len(boxes) > 0:
            for box in boxes:
                box[0] = int((box[0] / x_compression) + tile_offset)
                box[1] = int(((box[1] / y_compression - y_padding) / x_compression) + vertical_offset)
                box[2] = int((box[2] / x_compression) + tile_offset)
                box[3] = int(((box[3] / y_compression - y_padding) / x_compression) + vertical_offset)
            confidences = ["{:.3f}".format(confidence) for confidence in result.boxes.conf.tolist()]   # Confidence scores
            feature_type_ids = [str(int(cls)) for cls in result.boxes.cls.tolist()] # Feature IDs
            # Combine all information into a single list for each box
            for box, confidence, feature_type_id in zip(boxes, confidences, feature_type_ids):
                detection_info = [feature_type_id] + box + [confidence]
                predicted_boxes.append(detection_info)
    return predicted_boxes

def remove_overlapping_boxes(all_predictions: list):
    """
    Boxes of the same feature type and any overlap are removed.
    """

    def boxes_overlap(box1, box2):
        """
        Check if two boxes overlap and have the same feature type.
        """
        
        # Check if the feature types are the same
        if box1[0] != box2[0]:
            return False
        
        # Extract the coordinates of the boxes
        x1_min, y1_min, x1_max, y1_max = box1[1], box1[2], box1[3], box1[4]
        x2_min, y2_min, x2_max, y2_max = box2[1], box2[2], box2[3], box2[4]

        # Check if the boxes overlap   
        if x1_min < x2_max and x1_max > x2_min and y1_min < y2_max and y1_max > y2_min:
            return True
        return False

    def merge_boxes(box1, box2):
        """
        Merge two boxes into one.
        """
        x1_min, y1_min, x1_max, y1_max = box1[1], box1[2], box1[3], box1[4]
        x2_min, y2_min, x2_max, y2_max = box2[1], box2[2], box2[3], box2[4]
        
        merged_x_min = min(x1_min, x2_min)
        merged_y_min = min(y1_min, y2_min)
        merged_x_max = max(x1_max, x2_max)
        merged_y_max = max(y1_max, y2_max)
        merged_confidence = max(box1[5], box2[5])
        merged_feature_type = box1[0]  # Assuming feature types are the same
        
        return [merged_feature_type, merged_x_min, merged_y_min, merged_x_max, merged_y_max, merged_confidence]

    # Initialize a list to store non-overlapping boxes
    distinct_boxes = []
    
    # Sort predictions by confidence (higher confidence first)
    sorted_predictions = sorted(all_predictions, key=lambda x: x[5], reverse=True)
    
    while sorted_predictions:
        # Start with the box with the highest confidence
        current_box = sorted_predictions.pop(0)
        
        # Flag to check if we need to restart the overlap check
        merged = True
        
        while merged:
            merged = False
            for next_box in sorted_predictions[:]:
                if boxes_overlap(current_box, next_box):
                    current_box = merge_boxes(current_box, next_box)
                    sorted_predictions.remove(next_box)
                    merged = True
                    break  # Restart the overlap check with the new merged box
        
        # Add the non-overlapping (merged) box to distinct boxes
        distinct_boxes.append(current_box)
    
    return distinct_boxes

def analyse_image(image_path, model_path):

    """
    Main function includes:
    Preprocessing of image
    Analysis of processed images using object detection model
    Returns a list of all the predictions in the format [[Feature Type, Xmin, Ymin, Xmax, Ymax, Confidence]]
    """
     # Define parameters
    num_rakes = 10
    intesity_threshold_percentage = 20
    num_tiles = 5
    overlap_percentage = 20
    image_size = [640, 640]

    # Load the YOLO model
    model = YOLO(model_path)
    
    # Get image and convert to a NumPy array
    image = np.array(Image.open(image_path))

    image, vertical_offset = crop_image(image, num_rakes, intesity_threshold_percentage)
    tiled_images, tile_offsets = split_image(image, num_tiles, overlap_percentage)

    all_predictions = []
    for tile_index, (image, tile_offset) in enumerate(zip(tiled_images, tile_offsets)):
        image, x_compression = compress_in_x(image, image_size[1])
        image, y_compression, y_padding = make_square(image, image_size[0])

        # Call the model function
        results = model(image)

        """
        for r in results:
            im_array = r.plot()
            plt.imshow(im_array)
            plt.axis('off')
            plt.show()
        """
        # Format boundary boxes and convert coordinates to original image
        predicted_boxes = format_results(results, x_compression, y_compression, y_padding, vertical_offset, tile_offset)
        all_predictions.extend(predicted_boxes)
    # Remove overlapping boundary boxes
    all_predictions = remove_overlapping_boxes(all_predictions)
    return all_predictions

def write_to_db(all_predictions: list, database_path: str, image_path: str):

    """
    Writes resuts to the database
    """
    
    logging.info("Connecting to the database.")

    # Connect to SQLite database so we can write results
    con = sqlite3.connect(database_path)
    cur = con.cursor()
                            
    # Get the image ID from the database
    folderpath, filename = os.path.split(image_path)
    get_image_ID_command = "SELECT id FROM Image WHERE name = '" + filename + "'"
    logging.info(f"Get Image ID Command: {get_image_ID_command}")
    print("Get Image ID Command:", get_image_ID_command)
    res = cur.execute(get_image_ID_command)
    image_ID = (res.fetchone())[0]
                            
    # Write data about each box to the database
    for box in all_predictions:
                                
        # Other box properties
        box_class = box[0]
        x1 = box[1]
        y1 = box[2]
        x2 = box[3]
        y2 = box[4]
        box_conf = box[5]
                                
        # Timestamp
        now = datetime.now()
        timestamp = now.strftime("%m/%d/%Y %H:%M")
                                
        # Write in our data
        write_command = "INSERT INTO Feature (imageid, ftrType, x1, y1, x2, y2, confidence, date) VALUES (" + str(image_ID) + ", " + str(box_class) + ", " + str(x1) + ", " + str(y1) + ", " + str(x2) + ", " + str(y2) + ", " + str(box_conf) + ", '" + timestamp + "')"
        logging.info(f"Write Command: {write_command}")
        cur.execute(write_command)
        con.commit()

        logging.info("Successfully wrote data to the database.")



"""
#This code is for testing only. Comment out before compiling
# Compile using:
# pyinstaller --onefile --add-data ultralytics:ultralytics analysis_server_debugger.py


image_path = "C:/Users/J Gordon/Documents/RVHR/02 - Code - Copy/T1_PHP-VHR_20221114/T1_PHP-VHR_20221114_LeftRail/T1_PHP-VHR_LeftRail_20221114__0002410000.jpg"
database_path = "C:/Users/J Gordon/Documents/RVHR/02 - Code - Copy/T1_PHP-VHR_20221114/T1_PHP-VHR_20221114.db"
model_path = "C:/Users/J Gordon/Documents/RVHR/02 - Code - Copy/Analysis/best.pt"

# Check if paths exist and return error code if not
is_image_exist = os.path.exists(image_path)
is_model_exist = os.path.exists(model_path)
is_datab_exist = os.path.exists(database_path)
if is_image_exist == False or is_model_exist == False or is_datab_exist == False:
    print("Error with file paths")
else:
    all_predictions = analyse_image(image_path, model_path)
    if len(all_predictions) == 0:
        print("No features found")
    else:
        print("Features found! Writing to database.")
                            
        # Connect to SQLite database and write results
        new_image_path = image_path.replace("/", "\\")
        write_to_db(all_predictions, database_path, new_image_path)

"""

# Global variables
HOST = "127.0.0.1" # Server host location (locahost)

if __name__ == "__main__":
    
    # Input argument argument overrides for testing
    # Comment out before compiling
    #PORT = 65432
    
    # Executable input arguments
    # Uncomment before compiling
    PORT = int(sys.argv[1]) # Port to run server on
    
    # Open TCP listening socket on localhost and port defined in input argument
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        
        # Wait for connection on listening socket
        print("Waiting for connection from client...")
        logging.info("Waiting for connection from client...")
        s.bind((HOST, PORT))
        s.listen()
        conn, addr = s.accept()
        
        # Open connection socket
        with conn:
            print(f"Connected by {addr}")
            logging.info(f"Connected by {addr}")
            
            while True: # Continually reads data from the client
            
                # Decode received bytes data to string (assuming utf-8)
                data = conn.recv(1024)
                data = data.decode("utf-8")
                
                if not data:
                    break
                
                # Decipher command by reading first word before space
                prompt = data.split(";")
                command = prompt[0]
                
                # Analyse command received
                if command == "ANALYSE":
                    
                    # Get image, model, and database paths and remove quotation marks
                    image_path_untreated = prompt[1]
                    model_path_untreated = prompt[2]
                    database_path_untreated = prompt[3]
                    image_path = image_path_untreated[1:-1]
                    model_path = model_path_untreated[1:-1]
                    database_path = database_path_untreated[1:-1]
                    
                    # Check if paths exist and return error code if not
                    is_image_exist = os.path.exists(image_path)
                    is_model_exist = os.path.exists(model_path)
                    is_datab_exist = os.path.exists(database_path)
                    if is_image_exist == False or is_model_exist == False or is_datab_exist == False:
                        response_string = "003"
                        response = response_string.encode("utf-8")
                        conn.sendall(response)
                        logging.warning(f"Invalid paths: {image_path}, {model_path}, {database_path}")

                    
                    # Paths are valid so we can continue with analysis
                    else:
                        logging.info(f"Starting analysis of image: {image_path}")
                        all_predictions = analyse_image(image_path, model_path)
                        
                        # Return response 001 if no features were found
                        if len(all_predictions) == 0:
                            response_string = "001"
                            response = response_string.encode("utf-8")
                            conn.sendall(response)
                            logging.info("No features found in the analysis.")
                            
                        # If boxes were found, scale them up to original image size
                        else:
                            print("Features found! Writing to database.")
                            logging.info("Features found! Writing to database.")
                            
                            # Connect to SQLite database and write results
                            write_to_db(all_predictions, database_path, image_path)
                           
                            # Respond 002 indicating boxes were found
                            response_string = "002"
                            response = response_string.encode("utf-8")
                            conn.sendall(response)
                                                    
                # Stop command received
                elif command == "STOP":
                    print("Killing server and closing executable")
                    logging.info("Killing server and closing executable")
                    response_string = "006"
                    response = response_string.encode("utf-8")
                    conn.sendall(response)
                    break
                    
                # Invalid command received
                else:
                    print ("Invalid command received")
                    logging.warning("Invalid command received")
                    response_string = "005"
                    response = response_string.encode("utf-8")
                    conn.sendall(response)
