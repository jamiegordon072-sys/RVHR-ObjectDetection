# -*- coding: utf-8 -*-

# analysis_server.py

# Imports
import socket
import sys
import os
import numpy as np
from datetime import datetime
import torch
from ultralytics import YOLO
import sqlite3
import matplotlib.pyplot as plt

# Get the absolute path of the project directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Import from utils
from utils.preprocessing import open_image, crop_image, split_image, compress_in_x, make_square
from utils.inference import format_results, remove_overlapping_boxes
        

def analyse_image(image_path, model_path):
    # Define parameters
    num_tiles = 5
    image_size = [640, 640]

    # Load the YOLO model
    model = YOLO(model_path)

    # Check if GPU is available, otherwise use CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #print(f"Using device: {device}")

    preprocessed_images = []
    transformation_maps = []    

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

    # Call the model function
    results = model.predict(preprocessed_images, device=device)

    """
    for r in results:
        im_array = r.plot()
        plt.imshow(im_array)
        plt.axis('off')
        plt.show()
    """
    # Format results
    formatted_predictions = []
    for result, transformation_map in zip(results, transformation_maps):
        formatted_prediction = format_results(result, transformation_map)
        formatted_predictions.append(formatted_prediction)

    # Flatten predictions and filter out empty preds
    image_predictions = [bbox for pred in formatted_predictions[0:num_tiles] if pred for bbox in pred]
    
    # Remove overlapping boxes
    image_predictions = remove_overlapping_boxes(image_predictions)

    return image_predictions

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
        cur.execute(write_command)
        con.commit()



# This code is for testing only. Comment out before compiling
# Compile using:
# pyinstaller --onefile --add-data ultralytics:ultralytics inference_code/analysis_server.py

image_path =    "C:/Users/J Gordon/Documents/Rail Tech/RV-HR/Data/13 Rhp - Kh - Copy/241010-020838_RHP-KH_2_021-00009/RHP-KH_2_021-00009_20241010_00005_0,032km.jpg"
database_path = "C:/Users/J Gordon/Documents/Rail Tech/RV-HR/Data/13 Rhp - Kh - Copy/241010-020838_RHP-KH_2_021.db"
model_path = "best.pt"

# Check if paths exist and return error code if not
is_image_exist = os.path.exists(image_path)
is_model_exist = os.path.exists(model_path)
is_database_exist = os.path.exists(database_path)
if is_image_exist == False or is_model_exist == False or is_database_exist == False:
    print("Error with file paths")
else:
    all_predictions = analyse_image(image_path, model_path)
    if len(all_predictions) == 0:
        print("No features found")
    else:
        print("Features found! Writing to database.")
                            
        # Connect to SQLite database and write results
        new_image_path = image_path.replace("/", "\\")
        #write_to_db(all_predictions, database_path, new_image_path)

"""

# Global variables
HOST = "127.0.0.1" # Server host location (locahost)

if __name__ == "__main__":
    
    # Input argument argument overrides for testing
    # Comment out before compiling
    # PORT = 65432
    
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
                    
                    # Paths are valid so we can continue with analysis
                    else:
                        all_predictions = analyse_image(image_path, model_path)
                        
                        # Return response 001 if no features were found
                        if len(all_predictions) == 0:
                            response_string = "001"
                            response = response_string.encode("utf-8")
                            conn.sendall(response)
                            
                        # If boxes were found, scale them up to original image size
                        else:
                            print("Features found! Writing to database.")
                            
                            # Connect to SQLite database and write results
                            write_to_db(all_predictions, database_path, image_path)
                           
                            # Respond 002 indicating boxes were found
                            response_string = "002"
                            response = response_string.encode("utf-8")
                            conn.sendall(response)
                                                    
                # Stop command received
                elif command == "STOP":
                    print("Killing server and closing executable")
                    response_string = "006"
                    response = response_string.encode("utf-8")
                    conn.sendall(response)
                    break
                    
                # Invalid command received
                else:
                    print ("Invalid command received")
                    response_string = "005"
                    response = response_string.encode("utf-8")
                    conn.sendall(response)
"""