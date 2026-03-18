from ultralytics import YOLO
import torch

from models.detection import ImageData
from utils.utils import get_image, get_image_tag
from inference.postprocessing import insert_results
from utils.visualisation import display_features


def tile_inference(tile_path, model_path):
    """
    Perform inference on a single Rail Image TIle

    param image_path: Path to Rail Image Tile
    param model_path: Path to Model File
    returns: Predictions
    """
    
    # Load the Image
    tile_image = get_image(tile_path)
    tile_image_tag = get_image_tag(tile_path)
    tile = ImageData(
        image_tag= tile_image_tag,
        image=tile_image,
        detections=[]
    )

    # Load the YOLO model
    model = YOLO(model_path)

    # Call the model function
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = model.predict(tile_image, device=device)

    for result in results:
        tile = insert_results(tile, result)

    return tile
    
 

if __name__ == "__main__":  
    # Define Paths
    image_tile_path = "training_dataset/M3_T2_021-00009_20260303_00007_200,670km_xmin4198_xmax7462_aug0.jpg"
    model_path = "runs/detect/new dataset/weld_corrugation_microcorrugation_pit_blockjoint/weights/weld_corrugation_microcorrugation_pit_blockjoint.pt"

    tile_data = tile_inference(image_tile_path, model_path)
    display_features(tile_data)