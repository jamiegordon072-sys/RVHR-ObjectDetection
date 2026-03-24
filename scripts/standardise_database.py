import shutil
import os

from database.Database import RVHR_DB


# Define labels
LABELS = {
    0: "Weld",
    1: "Corrugation",
    2: "Pit",
    3: "Block Joint"
}


if __name__ == "__main__":
    original_db_path = "C:/Workspace/Rail Tech/RV-HR/08 - Sample Data/Training Runs/UNI-KHS/240429-041228_M1_T1.db"
    
    # Create training DB
    base, ext = os.path.splitext(original_db_path)
    training_db_path = f"{base}_training{ext}"
    shutil.copy2(original_db_path, training_db_path)
    print(f"Copied DB to: {training_db_path}")
    db = RVHR_DB(training_db_path)
    
    # Make Changes to DB
    db.reorder_feature_types(LABELS)    # Reorder Feature Type Table to match training labels
    db.combine_feature_type(ftr_type_src="Microcorrugation", ftr_type_dst="Corrugation")    # Convert Microcorrugation to Corrugation

    
    print("Training DB prepared successfully.")

