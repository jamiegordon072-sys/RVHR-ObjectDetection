import numpy as np
import cv2

def open_image(image_path):
    # Load image as BGR format using OpenCV
    return cv2.imread(image_path, cv2.IMREAD_COLOR)

def crop_image(image: np.ndarray, num_rakes: int = 10, intensity_threshold_percentage: float = 20):

    """
    Crops image down to width of the rail.
    Rail is detected by analyzing intensity changes along vertical rakes.
    
    :param image: Input image as a NumPy array (BGR format from OpenCV).
    :param num_rakes: Number of vertical rakes to analyze.
    :param intensity_threshold_percentage: Threshold for detecting rail boundaries.
    :return: Cropped image and vertical offset.
    """

    # Convert image to grayscale
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Get image dimensions
    image_height, image_width = grayscale_image.shape

    # Compute intensity threshold
    intensity_threshold = np.max(grayscale_image) * (intensity_threshold_percentage / 100)

    # Analyze vertical rakes across the image width
    lower_thresholds = []
    upper_thresholds = []
    for rake in range(num_rakes):
        # Determine rake position along image
        rake_index = int((rake + 1) * image_width/(num_rakes +1))

        # Extract the pixel intensity line from the image array
        pixel_intensity_line = grayscale_image[:, rake_index]

        # Find index where threshold is exceeded from top and bottom
        lower_threshold_index = np.argmax(pixel_intensity_line > intensity_threshold)
        upper_threshold_index = image_height - np.argmax(pixel_intensity_line[::-1] > intensity_threshold) -1

        lower_thresholds.append(lower_threshold_index)
        upper_thresholds.append(upper_threshold_index)

    # Compute average threshold positions
    lower_average = int(np.mean(lower_thresholds))
    upper_average = int(np.mean(upper_thresholds))

    # Crop image
    cropped_image = image[lower_average:upper_average, :]

    return cropped_image, (lower_average, upper_average)

def split_image(image: np.ndarray, num_tiles: int = 5, overlap_percentage: float = 20):
    """
    Splits the image in X into equal tiles with an overlap between adjacent tiles.
    The first and last tile will be smaller since they have overlap on only one side.

    :param image: Input image (NumPy array in OpenCV format).
    :param num_tiles: Number of tiles to divide the image into.
    :param overlap_percentage: Overlap percentage between adjacent tiles.
    :return: List of tiled images and their left-offsets.
    """

    # Get image dimensions
    image_height, image_width = image.shape[:2]

    # Calculate tile width
    tile_width = image_width // num_tiles

    # Calculate overlap in pixels
    overlap_px = int(tile_width * (overlap_percentage / 100))

    # Iterate over each tile
    tiled_images = []
    tile_boundaries = []
    for i in range(num_tiles):
        # Calculate the left and right boundaries of the current tile
        left = max(i * tile_width - overlap_px, 0)
        right = min((i + 1) * tile_width + overlap_px, image_width)

        # Extract the current tile from the input image
        tile = image[:, left:right]
        tiled_images.append(tile)
        tile_boundaries.append((left, right))

    return tiled_images, tile_boundaries

def compress_in_x(image: np.ndarray, target_width: int):
    """
    Compresses an image to a target width while maintaining aspect ratio.
    This scales the image proportionally in both X and Y directions.

    :param image: Input image as a NumPy array (OpenCV format).
    :param target_width: Desired width of the compressed image.
    :return: Resized image and the compression ratio.
    """

    # Get input image dimensions
    image_height, image_width = image.shape[:2]

    # Calculate the ratio of compression
    compression_ratio = target_width / image_width

    # Compute the new height with the same aspect ratio
    target_height = int(image_height * compression_ratio)
    
    # Resize the image
    compressed_image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
    
    return compressed_image, compression_ratio

def make_square(image: np.ndarray, target_height: int):
    """
    Adjusts an image to be square with a specified target height.
    
    - If the image height is greater than `target_height`, it is resized (compressed in Y).
    - If the image height is smaller than `target_height`, it is padded equally on top and bottom.

    :param image: Input image as a NumPy array (OpenCV format).
    :param target_height: Desired height/width of the square output image.
    :return: Square image, y compression ratio, and y padding value.
    """

    # Get current image dimensions
    image_height, image_width = image.shape[:2]

    # Case 1: Compress in Y direction (height > target_height)
    if image_height > target_height:
        y_compression = target_height / image_height
        y_padding = 0
        square_image = cv2.resize(image, (image_width, target_height), interpolation=cv2.INTER_AREA)

    # Case 2: Pad in Y direction (height < target_height)
    elif image_height < target_height:
        y_compression = 1
        y_padding = (target_height - image_height) // 2  # Equal padding on top and bottom

        # Add black padding (or change to any other color if needed)
        square_image = cv2.copyMakeBorder(image, y_padding, y_padding, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # Case 3: No changes required (height = target_height)
    else:
        y_compression = 1
        y_padding = 0
        square_image = image

    return square_image, y_compression, y_padding


def transform_annotations(annotations, transformation_map):
    """
    Adjust bounding box according to the transformation map to realign with the preprocessed image

    :param annoation: Original Bounding box annotation in YOLO format
    :param transformation_map: List of the relevant metrics used in the preprocessing of the image.
    :return: adjusted_annotation: Processed bounding box annotaion in YOLO format
    """

    # Get transformation metrics from transformation map
    (crop_y_min, crop_y_max), (tile_x_min, tile_x_max), compression_ratio, y_compression, y_padding = transformation_map

    # Crop in Y to width of rail
    y_crop_annotations = []
    for class_id, bbox in annotations:
        x_min, y_min, x_max, y_max = bbox

        # Check if the bounding box intersects with the cropped region
        if y_max > crop_y_min and y_min < crop_y_max:
            # Update the coordinates relative to the cropped image
            y_min = max(y_min - crop_y_min, 0)
            y_max = min(y_max - crop_y_min, crop_y_max - crop_y_min)
            y_crop_annotations.append((class_id, [x_min, y_min, x_max, y_max]))

    
    
    # Adjust the annotations for the current tile
    tile_annotations = []
    for class_id, bbox in y_crop_annotations:
        x_min, y_min, x_max, y_max = bbox
        # Check if the bounding box intersects with the current tile
        if x_min < tile_x_max and x_max > tile_x_min:
            # Adjust the coordinates relative to the tile
            x_min = max(x_min - tile_x_min, 0)
            x_max = min(x_max - tile_x_min, tile_x_max - tile_x_min)
            
            tile_annotations.append((class_id, [x_min, y_min, x_max, y_max]))

    # Process annotations for current tile
    processed_annotations = []
    for class_id, bbox in tile_annotations:
        x_min, y_min, x_max, y_max = bbox

        # Compress and or pad annotation
        x_min *= compression_ratio
        y_min = y_min * compression_ratio * y_compression + y_padding
        x_max *= compression_ratio
        y_max = y_max* compression_ratio * y_compression + y_padding

        processed_annotations.append((class_id, [x_min, y_min, x_max, y_max]))

    return processed_annotations
