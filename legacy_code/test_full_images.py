# Imports
import socket
import sys
import os
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageOps, ImageDraw, ImageFont
import cv2

       
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

def analyse_image(image_path, model):

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

def save_image(img_path: str, all_predictions: list, output_path: str):
    # Load the image
        image = Image.open(img_path)
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype("arial.ttf", size=100)

        # Iterate over all predictions
        for prediction in all_predictions:
            feature_type, xmin, ymin, xmax, ymax, confidence = prediction
            
            # Draw the bounding box
            draw.rectangle([xmin, ymin, xmax, ymax], outline="green", width=10)
            
            # Draw the label
            label = f"{feature_type} ({confidence})"
            text_location = (xmin, ymin-100)
            text_bbox = draw.textbbox(text_location, label, font=font)
            draw.rectangle(text_bbox, fill="green")
            draw.text(text_location, label, fill="black", font=font)
        
        # Save the image with the bounding boxes
        image.save(output_path)

# Load the YOLO model
model_path = "runs/detect/new dataset/weld_corrugation_microcorrugation_pit_blockjoint/weights/best.pt"
model = YOLO(model_path)

# Perform analysis on each file in the folder and save results
dir_path = "data/model_testing/general_test"
#dir_path = "C:/Users/J Gordon/Documents/RVHR/02 - Code - Copy/T1_PHP-VHR_20221114/T1_PHP-VHR_20221114_RightRail"
for path in os.listdir(dir_path):
    if os.path.isfile(os.path.join(dir_path, path)):
        img_path = dir_path + "/" + path
        output_path = dir_path + "/results/" + path
        all_predictions = analyse_image(img_path, model)
        save_image(img_path, all_predictions, output_path)