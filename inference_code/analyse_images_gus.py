# -*- coding: utf-8 -*-

# analysis_server.py

# Imports
import socket
import sys
import os
import shutil
import cv2
from datetime import datetime
from ultralytics import YOLO
import sqlite3


def delete_temp():
    
    """
    Deletes temp directory if it exists
    """
    
    isExist = os.path.exists(temp_path)
    if isExist:
        shutil.rmtree(temp_path)
        
        
def resize_image(image_path):
    
    """
    Resizes image to 640x640 for compatability with model.
    image_path: Path to image to be resized
    Returns path to resized image
    """
    
    # Resize specified image and save in temp directory
    im = cv2.imread(image_path)
    dim = (640, 640)
    resized = cv2.resize(im, dim, interpolation=cv2.INTER_AREA)
    now = datetime.now()
    resized_filename = now.strftime("%Y%d%m%H%M%S") # Use current date and time as filename accurate to second to avoid conflicts
    # resized_path = temp_path + "\\" + resized_filename + ".jpg"
    resized_path = temp_path + "\\" + resized_filename + ".jpg"
    cv2.imwrite(resized_path, resized)
    return resized_path


def get_boxes(resized_path, model_path):
    
    """
    Analyses a resized image with YOLOv8 model to get box info of detected features.
    resized_path: Path to a resized 640x640 image.
    model: Path to YOLOv8 model.
    Returns [[feature_class, x, y, w, h, confidence],...]
    Returns 0 if no features were detected
    """
    
    model = YOLO(model_path)
    
    results = model(resized_path)
    
    for r in results:
        
        r = r.numpy() # Convert to numpy array
        
        classes = r.boxes.cls # Numpy array of class IDs for each box
        coords = r.boxes.xyxy # Numpy array of xywh arrays for each box
        confs = r.boxes.conf # Numpy array of confidences for each box
        num_boxes = confs.size # Number of boxes
        
        all_boxes = [] # Initialise output list
        
        if num_boxes == 0:
            return 0
        else:
            counter = 0
            for i in classes:
                box_class = round(i)
                box_conf = confs[counter]
                box_xywh = coords[counter]
                box_x = box_xywh[0]
                box_y = box_xywh[1]
                box_w = box_xywh[2]
                box_h = box_xywh[3]
                this_box = [box_class, box_x, box_y, box_w, box_h, box_conf]
                all_boxes.append(this_box)
                counter += 1
            return all_boxes
        
        
def resize_box(xywh, larger_image_size):
    
    """
    Resizes xywh box for a 640x640 image to an image of a larger size.
    xywh: original xywh for a 640x640 image, as [x, y, w, h]
    larger_image_size: size of larger image (i.e. original size of 640x640 image before downscaling)
    as [width, height]
    Returns xywh of new box as [x, y, w, h]
    """
    
    w_im_small = 640
    h_im_small = 640
    w_im_large = larger_image_size[0]
    h_im_large = larger_image_size[1]
    x_small = xywh[0]
    y_small = xywh[1]
    w_small = xywh[2]
    h_small = xywh[3]
    
    x_large = round((w_im_large/w_im_small) * x_small)
    y_large = round((h_im_large/h_im_small) * y_small)
    w_large = round((w_im_large/w_im_small) * w_small)
    h_large = round((h_im_large/h_im_small) * h_small)
    
    return [x_large, y_large, w_large, h_large]


def get_image_size(image_path):
    
    """
    Gets size of image in pixels.
    image_path: Path to image to read size of.
    Returns list of integers, [width, height]
    """
    
    im = cv2.imread(image_path)
    height, width, channel = im.shape
    
    return [width, height]


# Global variables
HOST = "127.0.0.1" # Server host location (locahost)
localappdata = os.getenv('LOCALAPPDATA')
temp_path = localappdata + "\\KeyES\\temp\\images"


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
                prompt = data.split(",")
                command = prompt[0]
                
                # Analyse command received
                if command == "ANALYSE":
                    
                    # Get image, model, and database paths and remove quotation marks
                    image_path_untreated = prompt[1]
                    model_path_untreated = prompt[2]
                    datab_path_untreated = prompt[3]
                    image_path = image_path_untreated[1:-1]
                    model_path = model_path_untreated[1:-1]
                    datab_path = datab_path_untreated[1:-1]
                    
                    # Check if paths exist and return error code if not
                    is_image_exist = os.path.exists(image_path)
                    is_model_exist = os.path.exists(model_path)
                    is_datab_exist = os.path.exists(datab_path)
                    if is_image_exist == False or is_model_exist == False or is_datab_exist == False:
                        response_string = "003"
                        response = response_string.encode("utf-8")
                        conn.sendall(response)
                    
                    # Paths are valid so we can continue with analysis
                    else:
                    
                        # Delete temp directory if it exists
                        delete_temp()
                        
                        # Create new empty temp directory
                        os.makedirs(temp_path)
                    
                        # Resize image and store in temp in folder
                        resized_path = resize_image(image_path)
                        
                        # Analyse resized image
                        all_boxes = get_boxes(resized_path, model_path)
                        
                        # Return response 001 if no features were found
                        if all_boxes == 0:
                            response_string = "001"
                            response = response_string.encode("utf-8")
                            conn.sendall(response)
                            
                        # If boxes were found, scale them up to original image size
                        else:
                            new_boxes = []
                            for box in all_boxes:
                                box_class = box[0]
                                box_conf = box[5]
                                old_x = box[1]
                                old_y = box[2]
                                old_w = box[3]
                                old_h = box[4]
                                old_xywh = [old_x, old_y, old_w, old_h]
                                original_size = get_image_size(image_path)
                                new_xywh = resize_box(old_xywh, original_size)
                                new_x = new_xywh[0]
                                new_y = new_xywh[1]
                                new_w = new_xywh[2]
                                new_h = new_xywh[3]
                                new_box = [box_class, new_x, new_y, new_w, new_h, box_conf]
                                new_boxes.append(new_box)
                                
                            print("Features found! Writing to database.")
                            
                            # Connect to SQLite database so we can write results
                            con = sqlite3.connect(datab_path)
                            cur = con.cursor()
                            
                            # Get the image ID from the database
                            folderpath, filename = os.path.split(image_path)
                            get_image_ID_command = "SELECT id FROM Image WHERE folderpath='" + folderpath + "' AND name = '" + filename + "'"
                            res = cur.execute(get_image_ID_command)
                            image_ID = (res.fetchone())[0]
                            
                            # Write data about each box to the database
                            for box in new_boxes:
                                
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
                            
                            # Respond 002 indicating boxes were found and written to database
                            response_string = "002"
                            response = response_string.encode("utf-8")
                            conn.sendall(response)
                                
                        # Delete temp directory
                        delete_temp()
                    
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