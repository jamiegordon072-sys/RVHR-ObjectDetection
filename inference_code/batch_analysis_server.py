# -*- coding: utf-8 -*-

# batch_analysis_server.py

"""
!!! FULL BATCH ANALYSIS IS NOT POSSIBLE SINCE ENCOUNTER MEMORY ISSUES WITH RESULTS !!!

Instead can divide into mini batches of (eg. 10) images
To accomodate this the code accepts a list of image paths rather than a folder path
"""

# Imports
import sys
import os
import numpy as np
import time
import socket
import json
import torch
from datetime import datetime
from ultralytics import YOLO
import sqlite3


# Get the absolute path of the project directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Import from utils
from utils.preprocessing import open_image, crop_image, split_image, compress_in_x, make_square
from utils.inference import format_results, remove_overlapping_boxes


def analyse_images(image_paths, model_path):

    """
    Main function includes:
    Preprocessing of image
    Analysis of processed images using object detection model
    Returns a list of all the predictions in the format [[Feature Type, Xmin, Ymin, Xmax, Ymax, Confidence]]
    """


    # Define preprocessing parameters
    num_tiles = 5
    image_size = [640, 640]

    # Start preprocessing timer
    preprocess_start = time.perf_counter()

    # Preprocess Images
    preprocessed_images = []
    transformation_maps = []
    for image_path in image_paths:
        # Load image
        image = open_image(image_path)

        image, (y_crop_min, _) = crop_image(image)
        tiled_images, tile_boundaries = split_image(image, num_tiles)
        
        for image, (tile_x_offset, _) in zip(tiled_images, tile_boundaries):
            image, x_compression = compress_in_x(image, image_size[1])
            image, y_compression, y_padding = make_square(image, image_size[0])

            transformation_map = [x_compression, y_compression, y_padding, y_crop_min, tile_x_offset]

            preprocessed_images.append(image)
            transformation_maps.append(transformation_map)
    
    # End preprocessing timer
    preprocess_end = time.perf_counter()
    print(f"✅ Preprocessing Completed! Time Taken: {preprocess_end - preprocess_start:.4f} sec")

    # Load the YOLO model
    model = YOLO(model_path)

    # Check if GPU is available, otherwise use CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Start inference timer
    inference_start = time.perf_counter()

    # Call the model function
    results = model.predict(preprocessed_images, device=device)

    # End model loading timer
    inference_end = time.perf_counter()
    print(f"✅ Inference Completed using {device}! Time Taken: {inference_end - inference_start:.4f} sec")

    
    # Format results
    formatted_predictions = []
    for result, transformation_map in zip(results, transformation_maps):
        formatted_prediction = format_results(result, transformation_map)
        formatted_predictions.append(formatted_prediction)

    # Group together tiles for the same image
    all_predictions = []
    for i, image_path in zip(range(0, len(formatted_predictions), num_tiles), image_paths):
        # Flatten predictions and filter out empty preds
        #image_predictions = [bbox for pred in formatted_predictions[0:num_tiles] if pred for bbox in pred]
        image_predictions = [bbox for pred in formatted_predictions[i:i + num_tiles] if pred for bbox in pred]

        image_predictions = remove_overlapping_boxes(image_predictions)
        all_predictions.append(image_predictions)

    return all_predictions
        
def write_to_db(all_predictions: list, database_path: str, image_path: str):

    """
    Writes resuts to the database
    """
    
    # Connect to SQLite database so we can write results
    con = sqlite3.connect(database_path)
    cur = con.cursor()
                            
    # Get the image ID from the database
    folderpath, filename = os.path.split(image_path)
    get_image_ID_command = "SELECT id FROM Image WHERE name = '" + filename + "'"
    res = cur.execute(get_image_ID_command)
    image_ID = (res.fetchone())[0]

    # Timestamp
    now = datetime.now()
    timestamp = now.strftime("%d/%m/%Y %H:%M")

    # Write data about each box to the database
    for box in all_predictions:
                                
        # Other box properties
        box_class = box[0]
        x1 = box[1]
        y1 = box[2]
        x2 = box[3]
        y2 = box[4]
        box_conf = box[5]
                                                               
        # Write in our data
        write_command = "INSERT INTO Feature (imageid, ftrType, x1, y1, x2, y2, confidence, date) VALUES (" + str(image_ID) + ", " + str(box_class) + ", " + str(x1) + ", " + str(y1) + ", " + str(x2) + ", " + str(y2) + ", " + str(box_conf) + ", '" + timestamp + "')"
        cur.execute(write_command)
        
    write_command = "UPDATE Image SET analysed=1, analysedDate= '" + timestamp + "' WHERE name = '" + filename + "'"
    cur.execute(write_command)
    con.commit()


"""

# This code is for testing only. Comment out before compiling
# Compile using:
# pyinstaller --onefile --add-data ultralytics:ultralytics --add-data "utils:utils" inference_code/batch_analysis_server.py

def load_json(json_file):
    #Load JSON from file, if it exists.
    try:
        with open(json_file, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"⚠ JSON file '{json_file}' not found. Switching to manual input.")
    except json.JSONDecodeError as e:
        print(f"Error loading JSON: {e}")
    return None

# Load inputs from command text or manual inputs
command_path = "inference_code/Example Batch Analysis Command.txt"
request = load_json(command_path)
if request:
    image_paths = request.get("Image Paths", [])
    database_path = request.get("Database Path", "")
    model_path = request.get("Model Path", "")
else:
    image_paths = ["C:/Users/J Gordon/Documents/Rail Tech/RV-HR/Data/13 Rhp - Kh - Copy/241010-020838_RHP-KH_2_021-00009/RHP-KH_2_021-00009_20241010_00005_0,032km.jpg"]
    database_path = "C:/Users/J Gordon/Documents/Rail Tech/RV-HR/Data/13 Rhp - Kh - Copy/241010-020838_RHP-KH_2_021.db"
    model_path = "C:/Users/J Gordon/Documents/Rail Tech/RV-HR/Code/Analysis/best.pt"

# Check if paths exist and return error code if not
missing_images = [img for img in image_paths if not os.path.isfile(img)]
is_model_exist = os.path.isfile(model_path)
is_database_exist = os.path.isfile(database_path)
if missing_images or is_model_exist == False or is_database_exist == False:
    print("Error with file paths")
else:
    all_predictions = analyse_images(image_paths, model_path)
    if len(all_predictions) == 0:
        print("No Features Found")
    else:
        print("Writing Features to Database:",
        all_predictions)
    

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
        s.bind((HOST, PORT))
        s.listen()
        conn, addr = s.accept()
        
        # Open connection socket
        with conn:
            print(f"Connected by {addr}")
            
            while True: # Continually reads data from the client
            
                # Decode received bytes data to string (assuming utf-8)
                data = conn.recv(8192).decode("utf-8")
                
                if not data:
                    break
                
                # Try Parsing JSON
                try:
                    request = json.loads(data)
                except json.JSONDecodeError as e:
                    print(f"JSON decode error: {e}")
                    conn.sendall("003".encode("utf-8"))
                    continue  # Skip this iteration and wait for the next valid message
                
                command = request.get("Command", "").upper()
                
                if command == "ANALYSE":
                    try:
                        image_paths = request.get("Image Paths", [])
                        database_path = request.get("Database Path", "")
                        model_path = request.get("Model Path", "")
                        
                        # Check if paths exist and return error code if not
                        missing_images = [img for img in image_paths if not os.path.isfile(img)]
                        is_model_exist = os.path.isfile(model_path)
                        is_database_exist = os.path.isfile(database_path)
                        if missing_images:
                            print(f"Missing Images: {missing_images}")
                            conn.sendall("005".encode("utf-8"))
                        elif not is_model_exist:
                            print(f"Model Missing: {model_path}")
                            conn.sendall("006".encode("utf-8"))
                        elif not is_database_exist:
                            print(f"Database Missing: {database_path}")
                            conn.sendall("007".encode("utf-8"))
                        
                        # Paths are valid. Continue with analysis
                        else:
                            try:
                                print("Beginning Analysis")
                                all_predictions = analyse_images(image_paths, model_path)
                            except Exception as e:
                                print(f"Inference error: {e}")
                                conn.sendall("008".encode("utf-8"))  # Model inference failed
                                continue
                                
                            
                            # If features found then write to db
                            if any(len(preds) > 0 for preds in all_predictions):
                                try:
                                    for image_predictions, image_path in zip(all_predictions, image_paths):
                                        # Connect to SQLite database and write results
                                        write_to_db(image_predictions, database_path, image_path)
                                    # Respond 002 indicating boxes were found
                                    print("Features found! Writing to database.")
                                    conn.sendall("002".encode("utf-8"))
                                except Exception as e:
                                    print(f"Database write error: {e}")
                                    conn.sendall("009".encode("utf-8"))
                            
                            # Return response 001 if no features were found   
                            else:
                                print("No Features Found")
                                conn.sendall("001".encode("utf-8"))
                                
                    except Exception as e:
                        print(f"Unexpected error with analysis: {e}")
                        conn.sendall("010".encode("utf-8"))
                
                                                    
                # Stop command received
                elif command == "STOP":
                    print("Killing server and closing executable")
                    conn.sendall("011".encode("utf-8"))
                    break
                    
                # Invalid command received
                else:
                    print ("Invalid command received")
                    conn.sendall("004".encode("utf-8"))
