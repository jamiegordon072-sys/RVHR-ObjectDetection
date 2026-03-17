from datetime import datetime

def format_results(results: list, transformation_map: list):

    """
    Results are put in correct format.
    The results from the prepocessed tiled images are converted into the coordinate system of the original full size image.
    The results for the different tiles are compiled into a single variable.
    Each result are put in the format [Feature Type, Xmin, Ymin, Xmax, Ymax, Confidence]
    The transformation map denotes how to convert back to the original coordintae system.
    It has the format [x_compression, y_compression, y_padding, vertical_offset, tile_offset]
    """
    x_compression, y_compression, y_padding, vertical_offset, tile_offset = transformation_map
    predicted_boxes = []

    # Timestamp
    now = datetime.now()

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
                if int(feature_type_id) >= 2: # Remove Microcorrugation - Decrement all subsequent feature type IDs by 1
                    feature_type_id = str(int(feature_type_id) - 1)
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
        merged_confidence = max(box1[5], box2[5]) # Take max confidence
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
