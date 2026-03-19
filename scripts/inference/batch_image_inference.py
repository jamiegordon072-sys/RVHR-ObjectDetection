from src.database.Database import RVHR_DB
from src.inference.inference import batch_inference
from src.utils.visualisation import display_features


if __name__ == "__main__":
    image_paths = [
        "C:/Workspace/Rail Tech/RV-HR/Data/CPH 032026/260303-025143_M3_T2_LeftRail/M3_T2_021-00009_20260303_00007_200,670km.jpg",
        "C:/Workspace/Rail Tech/RV-HR/Data/CPH 032026/260303-025143_M3_T2_LeftRail/M3_T2_021-00009_20260303_00014_200,684km.jpg"
    ]
    model_path = "runs/detect/new dataset/weld_corrugation_microcorrugation_pit_blockjoint/weights/weld_corrugation_microcorrugation_pit_blockjoint.pt"
    db_path = "C:/Workspace/Rail Tech/RV-HR/Data/CPH 032026/260303-025143_M3_T2.db"

    batch_image_data = batch_inference(image_paths, model_path)

    db = RVHR_DB(db_path)

    for image_data in batch_image_data:
        display_features(image_data)
        db.insert_detections(image_data)
