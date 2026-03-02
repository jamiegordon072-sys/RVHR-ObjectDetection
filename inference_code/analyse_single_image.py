from ultralytics import YOLO
import cv2
from datetime import datetime
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys

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
    print(f"Using device: {device}")
    
     # Load image
    image = open_image(image_path)

    image, vertical_offset = crop_image(image)
    tiled_images, tile_offsets = split_image(image, num_tiles)

    preprocessed_images = []
    transformation_maps = []
    for result_index, (image, tile_offset) in enumerate(zip(tiled_images, tile_offsets)):
        image, x_compression = compress_in_x(image, image_size[1])
        image, y_compression, y_padding = make_square(image, image_size[0])

        transformation_map = [x_compression, y_compression, y_padding, vertical_offset, tile_offset]

        preprocessed_images.append(image)
        transformation_maps.append(transformation_map)

    # Call the model function
    results = model.predict(preprocessed_images, device=device)

    """"""
    for r in results:
        im_array = r.plot()
        plt.imshow(im_array)
        plt.axis('off')
        plt.show()
    
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


model_path = "C:/Users/J Gordon/Documents/Rail Tech/RV-HR/Code/RVHR 1.0.9/Analysis/best.pt"
image_path = "C:/Users/J Gordon/Downloads/RV/250826-204807_Mainline_1_021_LeftRail/Mainline_1_021-00009_20250826_00006_11,025km.jpg"
image_path = "C:/Users/J Gordon/Documents/Rail Tech/RV-HR/Data/4 FB-FOR/240429-025515_M1+M2_T1_LeftRail/M1+M2_T1_LeftRail_20240429_00038_3,234km.jpg"
image_path = "C:/Users/J Gordon/Downloads/RV/250826-204807_Mainline_1_021_RightRail/Mainline_1_021-00008_20250826_00019_11,051km.jpg"
all_predictions = analyse_image(image_path, model_path)
print(all_predictions)

"""

if __name__ == "__main__":  
    # Extract file paths from command-line arguments
    model_path = sys.argv[1]
    image_path = sys.argv[2]

    all_predictions = analyse_image(image_path, model_path)
    print(all_predictions)
"""