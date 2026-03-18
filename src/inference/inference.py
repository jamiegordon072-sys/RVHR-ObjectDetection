from ultralytics import YOLO
import torch

from models.detection import ImageData
from preprocessing.preprocess import preprocess_image
from inference.postprocessing import insert_results, combine_tiles, handle_overlaps


def image_inference(image_path: str, model_path: str) -> ImageData:
    """
    Perform inference on a single Rail Image

    param image_path: Path to Rail Image
    param model_path: Path to Model File
    returns: Image Data object containing the predictions
    """
    

    # Preprocess Image
    tiles, trans_maps = preprocess_image(image_path)
    tile_images = [tile.image for tile in tiles]

    # Load the YOLO model
    model = YOLO(model_path)

    # Call the model function
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = model.predict(tile_images, device=device)
    
    # Format results
    tile_results = []
    for tile, result in zip(tiles, results):
        tile_result = insert_results(tile, result)
        tile_results.append(tile_result)
    
    image_object = combine_tiles(image_path, tile_results, trans_maps)
    image_object = handle_overlaps(image_object)

    return image_object


def batch_inference(image_paths: list[str], model_path: str) -> list[ImageData]:
    """
    Perform batch inference on a list of images

    params image_paths: List of Image Paths
    param model_path: Path to Model
    """

    # Preprocess Images
    tiles_batch = []
    tile_images_batch = []
    trans_maps_batch = []
    for image_path in image_paths:
        tiles, image_trans_maps = preprocess_image(image_path)
        tile_images = [tile.image for tile in tiles]
        tiles_batch.extend(tiles)
        tile_images_batch.extend(tile_images)
        trans_maps_batch.extend(image_trans_maps)
    # Load the YOLO model
    model = YOLO(model_path)

    # Check if GPU is available, otherwise use CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = model.predict(tile_images_batch, device=device)
    
    # Format results
    tile_results = []
    for tile, result in zip(tiles_batch, results):
        tile_result = insert_results(tile, result)
        tile_results.append(tile_result)

    # Group together tiles for the same image
    tiles_per_image = int(len(tile_results)/len(image_paths))
    
    # Cluster tile results into their original source images
    image_objects = []
    for i, image_path in zip(range(0, len(tile_results), tiles_per_image), image_paths):
        # Get results and trans maps corresponding to current image
        image_tile_results = tile_results[i:i+tiles_per_image]
        image_trans_maps = trans_maps_batch[i:i+tiles_per_image]

        # Combine tile results and handle overlaps
        image_object = combine_tiles(image_path, image_tile_results, image_trans_maps)
        image_object = handle_overlaps(image_object)

        image_objects.append(image_object)
    
    return image_objects

