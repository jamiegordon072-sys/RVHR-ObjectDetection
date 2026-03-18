from models.detection import Detection, ImageData
from models.transformation import TransformationMap
from utils.utils import get_image, get_image_tag


def insert_results(image: ImageData, result) -> ImageData:
    """
    Extract the list of detections from the YOLO model output and insert into the Image Object

    param results: ImageData Object to place the detections into
    param results: Results from the YOLO model inference
    returns: Image Data containing the detections
    """

    detections = []
    bboxes = result.boxes.xyxy.tolist()
    if bboxes is not None:
        labels = result.boxes.cls.tolist()
        confs = result.boxes.conf.tolist()
        for bbox, label, conf in zip(bboxes, labels, confs):
            detection = Detection(
                label = int(label),
                bbox = [int(x) for x in bbox],
                confidence = "{:.3f}".format(conf)
            )
            detections.append(detection)
    
    image.detections = detections

    return image


def combine_tiles(image_path: str, tiles: list[ImageData], trans_maps: list[TransformationMap]) -> ImageData:
    """
    Combine Tile Results into Single Image Results Object. Use Transformation Maps for each tile to convert to global coords.

    param tiles: List of Image Tile Objects
    param trans_maps: List of Transformation Maps for each Tile
    returns: Image Object containing the Detections across all Tiles
    """

    # Load image from path
    image = get_image(image_path)
    image_tag = get_image_tag(image_path)

    image_detections =[]
    for tile_object, trans_map in zip(tiles, trans_maps):
        for detection in tile_object.detections:
            # Convert tile bbox to global coordinates
            bbox = detection.bbox
            global_bbox=[
                int((bbox[0] / trans_map.x_compression) + trans_map.tile_x_min),
                int(((bbox[1] / trans_map.y_compression - trans_map.y_padding) / trans_map.x_compression) + trans_map.crop_y_min),
                int((bbox[2] / trans_map.x_compression) + trans_map.tile_x_min),
                int(((bbox[3] / trans_map.y_compression - trans_map.y_padding) / trans_map.x_compression) + trans_map.crop_y_min)
            ] 
            # Recreate detection object with new global bbox
            image_detection = Detection(
                label=detection.label,
                bbox=global_bbox,
                confidence=detection.confidence
            )

            image_detections.append(image_detection)
    
    image_object = ImageData(
        image_tag = image_tag,
        image = image,
        detections = image_detections
    )

    return image_object
    

def boxes_overlap(det1: Detection, det2: Detection):
    """
    Check if two boxes overlap and have the same feature type.
    """
    
    # Check if the feature types are the same
    if det1.label != det2.label:
        return False
    
    # Extract the coordinates of the boxes
    x1_min, y1_min, x1_max, y1_max = det1.bbox
    x2_min, y2_min, x2_max, y2_max = det2.bbox

    # Check if the boxes overlap   
    if x1_min < x2_max and x1_max > x2_min and y1_min < y2_max and y1_max > y2_min:
        return True
    return False


def merge_boxes(det1: Detection, det2: Detection) -> Detection:
    """
    Merge two boxes into one.
    """
    x1_min, y1_min, x1_max, y1_max = det1.bbox
    x2_min, y2_min, x2_max, y2_max = det2.bbox
    
    merged_x_min = min(x1_min, x2_min)
    merged_y_min = min(y1_min, y2_min)
    merged_x_max = max(x1_max, x2_max)
    merged_y_max = max(y1_max, y2_max)
    merged_confidence = max(det1.confidence, det2.confidence) # Take max confidence
    merged_label = det1.label  # Can Assume feature types are the same

    merged_detection = Detection(
        label = merged_label,
        bbox = [
            merged_x_min,
            merged_y_min,
            merged_x_max,
            merged_y_max
        ],
        confidence = merged_confidence
    )

    return merged_detection


def handle_overlaps(image: ImageData) -> ImageData:
    """
    Remove Overlapping Boxes of the Same Feature Type

    param image: Image Object containing all detections
    returns: Image Object with overlapping detections removed
    """

    detections = image.detections

    # Initialize a list to store non-overlapping boxes
    distint_detections = []
    
    # Sort predictions by confidence (higher confidence first)
    sorted_detections = sorted(
        detections,
        key=lambda d: d.confidence,
        reverse=True)
    
    while sorted_detections:
        # Start with the box with the highest confidence
        current_detection = sorted_detections.pop(0)
        
        # Flag to check if we need to restart the overlap check
        merged = True
        
        while merged:
            merged = False
            for next_detection in sorted_detections[:]:
                if boxes_overlap(current_detection, next_detection):
                    current_detection = merge_boxes(current_detection, next_detection)
                    sorted_detections.remove(next_detection)
                    merged = True
                    break  # Restart the overlap check with the new merged box
        
        # Add the non-overlapping (merged) box to distinct boxes
        distint_detections.append(current_detection)
    
    image.detections = distint_detections
    
    return image

