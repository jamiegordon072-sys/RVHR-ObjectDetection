from src.database.Database import RVHR_DB
from src.inference.inference import image_inference
from src.utils.visualisation import display_features


if __name__ == "__main__":  
    # Define Paths
    image_path = "C:/Workspace/Rail Tech/RV-HR/Data/CPH 032026/260303-025143_M3_T2_LeftRail/M3_T2_021-00009_20260303_00007_200,670km.jpg"
    model_path = "runs/detect/new dataset/weld_corrugation_microcorrugation_pit_blockjoint/weights/weld_corrugation_microcorrugation_pit_blockjoint.pt"
    db_path = "C:/Workspace/Rail Tech/RV-HR/Data/CPH 032026/260303-025143_M3_T2.db"

    image_data = image_inference(image_path, model_path)
    display_features(image_data)

    db = RVHR_DB(db_path)
    db.insert_detections(image_data)
