# -*- coding: utf-8 -*-

# batch_analysis_server.py

# Compile using:
# pyinstaller --onefile --add-data ultralytics:ultralytics --paths src scripts/inference/batch_inference_server.py

"""
!!! FULL BATCH ANALYSIS IS NOT POSSIBLE SINCE ENCOUNTER MEMORY ISSUES WITH RESULTS !!!

Instead can divide into mini batches of (eg. 10) images
To accomodate this the code accepts a list of image paths rather than a folder path
"""

# Imports
import sys
import os
import socket
import json

from database.Database import RVHR_DB
from inference.inference import batch_inference


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
                        db_path = request.get("Database Path", "")
                        model_path = request.get("Model Path", "")
                        
                        # Check if paths exist and return error code if not
                        missing_images = [img for img in image_paths if not os.path.isfile(img)]
                        is_model_exist = os.path.isfile(model_path)
                        is_database_exist = os.path.isfile(db_path)
                        if missing_images:
                            print(f"Missing Images: {missing_images}")
                            conn.sendall("005".encode("utf-8"))
                        elif not is_model_exist:
                            print(f"Model Missing: {model_path}")
                            conn.sendall("006".encode("utf-8"))
                        elif not is_database_exist:
                            print(f"Database Missing: {db_path}")
                            conn.sendall("007".encode("utf-8"))
                        
                        # Paths are valid. Continue with analysis
                        else:
                            try:
                                print("Beginning Analysis")
                                batch_image_data = batch_inference(image_paths, model_path)
                            except Exception as e:
                                print(f"Inference error: {e}")
                                conn.sendall("008".encode("utf-8"))  # Model inference failed
                                continue
                                
                            
                            # If features found then write to db
                            if any(len(image_data.detections) > 0 for image_data in batch_image_data):
                                try:
                                    for image_data in batch_image_data:
                                        # Connect to SQLite database and write results
                                        db = RVHR_DB(db_path)
                                        db.insert_detections(image_data)
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
