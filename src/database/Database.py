import os
import math
import sqlite3
from datetime import datetime

from models.detection import Detection, ImageData

class DB:
    """
    Class Data:
        filepath - filepath of the database
        name - database file name
        conn - sqlite database connection

    Methods:
        select - execute sql select statement and return values
        update - execute sql update statement
        update_many - execute sql update statement many times
        insert - execute sql insert statement
        insert_many - execute sql insert statement many times
    """
    def __init__(self, filepath: str):
        self.filepath: str = filepath
        self.name: str = os.path.split(filepath)[1]
        self.conn = sqlite3.connect(filepath)


    def __del__(self):
        self.conn.commit()  # commit changes to database
        self.conn.close()  # close database connection


    def _select(self, select_statement: str, preview: bool = False) -> list:
        """
        Selects values from database

        :param select_statement: sql select statement string
        :param preview: print preview of data returned
        :return values from database
        """
        cur = self.conn.cursor()
        cur.execute(select_statement)
        res = cur.fetchall()
        if preview:
            for i in range(min(5, len(res))):
                print(res[i])
        cur.close()
        return res
    

    def _update(self, update_statement: str, values: tuple):
        """
        Executes an UPDATE statement

        :param update_statement: SQL update statement with placeholders (?)
        :param values: tuple of values
        """
        cur = self.conn.cursor()
        cur.execute(update_statement, values)
        self.conn.commit()
        cur.close()


    def _update_many(self, update_statement: str, values: list[tuple]):
        cur = self.conn.cursor()
        cur.executemany(update_statement, values)
        self.conn.commit()
        cur.close()
    

    def _insert(self, insert_statement: str, values: tuple):
        """
        Inserts values into database

        :param insert_statement: SQL insert statement with placeholders (?)
        :param values: tuple of values to insert
        """
        cur = self.conn.cursor()
        cur.execute(insert_statement, values)
        self.conn.commit()
        cur.close()


    def _insert_many(self, insert_statement: str, values: list[tuple]):
        cur = self.conn.cursor()
        cur.executemany(insert_statement, values)
        self.conn.commit()
        cur.close()
    

class RVHR_DB(DB):
    """
    Database class for RVHR. Table names are "Image, "Feature, "FeatureType"
    """

    def get_img_id(self, img_name: str) -> int:
        """
        Get image id from database given image name
        :param img_name: name of the image file
        :return: image id
        """
        select_statement = f"""
            SELECT id
            FROM Image
            WHERE name='{img_name}'
        """
        res = self._select(select_statement)
        if len(res) == 0:
            raise ValueError(f"No image with name {img_name} found in database")
        return res[0][0]
    
    def get_img_name(self, img_id: int) -> str:
        """
        Get image name from database given image id
        :param img_id: image id
        :return: image name
        """
        select_statement = f"""
            SELECT name
            FROM Image
            WHERE id={img_id}
        """
        res = self._select(select_statement)
        if len(res) == 0:
            raise ValueError(f"No image with id {img_id} found in database")
        return res[0][0]
    
    def get_labelled_img_ids(self, ftr_types) -> list[int]:
        """
        Get image ids that have valid annotations (ie. status=1, conf=1; ftr_type in provided list)
        param ftr_types: list of feature types to filter by
        return: list of image ids that have valid annotations
        """

        ftr_type_str = ",".join(str(x) for x in ftr_types)

        select_statement = f"""
            SELECT DISTINCT imageid
            FROM Feature
            WHERE status = 1
            AND confidence = 1
            AND ftrType IN ({ftr_type_str})
        """
        res = self._select(select_statement)
        return [row[0] for row in res]

    def get_labelled_features(self, img_id: int) -> list[Detection]:
        """
        Get list of detections for a given image id
        :param img_id: image id
        :return: list of Detection objects
        """
        select_statement = f"""
            SELECT ftrType, x1, y1, x2, y2
            FROM Feature
            WHERE imageid={img_id}
            AND status=1
            AND confidence=1
        """
        res = self._select(select_statement)

        annotations = []
        for row in res:
            label, x_min, y_min, x_max, y_max = row
            annotation = Detection(
                label = label,
                bbox = [x_min, y_min, x_max, y_max]
            )
            annotations.append(annotation)
        return annotations
            
    
    def insert_detections(self, image_data: ImageData) -> None:
        """
        Write a list of detections to the Feature Table in the DB

        param image_path: Path to the Image
        param detections: List of Detections
        """

        # Get Image ID
        image_name = image_data.image_tag + ".jpg"
        img_id = self.get_img_id(image_name)

        # Get Timestamp
        now = datetime.now()
        timestamp = now.strftime("%d/%m/%Y %H:%M")

        # Insert detections into Feature Table
        insert_statement = f"""
            INSERT INTO Feature
            (imageid, ftrType, x1, y1, x2, y2, confidence, date)
            VALUES
            (?, ?, ?, ?, ?, ?, ?, ?)
        """    
        insert_values_list = [
            (img_id, d.label, *d.bbox, d.confidence, timestamp)
            for d in image_data.detections
        ]
        self._insert_many(insert_statement, insert_values_list)

        # Mark Image as analysed
        update_statement = f"""
            UPDATE Image
            SET
            analysed = 1,
            analysedDate= ?
            WHERE id =?
        """
        update_values = (timestamp, img_id)
        self._update(update_statement, update_values)
    
    
    def reorder_feature_types(self, labels: dict[int, str]) -> None:
        """
        Reorder feature type list to match given labels dict

        This will:
        1. Assign new IDs to known labels (e.g. Weld → 0, etc.)
        2. Assign remaining feature types sequential IDs after those
        3. Safely remap IDs using a temporary offset to avoid collisions
        4. Update foreign keys in the Feature table accordingly

        param labels: Dictionary containing the desired feature type order
        """
        
        # Step 1: Get current FeatureType table
        select_statement = f"""
            SELECT id, name
            FROM FeatureType
        """
        res = self._select(select_statement)
        current_name_to_id = {name: fid for fid, name in res}

        # Step 2: Insert any missing labels
        insert_statement = f"""
            INSERT INTO FeatureType
            (name)
            VALUES
            (?)
        """
        for _, name in labels.items():
            if name not in current_name_to_id:
                self._insert(insert_statement, (name,))
                select_statement = f"""
                    SELECT id
                    FROM FeatureType
                    WHERE name = "{name}"
                """
                new_id = self._select(select_statement)
                current_name_to_id[name] = new_id

        # Step 3: Re-fetch the table including newly inserted labels
        select_statement = f"""
            SELECT id, name
            FROM FeatureType
        """
        res = self._select(select_statement)

        # Step 4: Assign new IDs
        # Track used IDs to avoid collisions
        used_ids = set(labels.keys())
        name_to_new_id = {name: fid for fid, name in labels.items()}

        # Start assigning remaining IDs after max label ID
        next_id = max(labels.keys()) + 1
        for old_id, name in res:
            if name not in name_to_new_id:
                # Skip IDs already in used_ids
                while next_id in used_ids:
                    next_id += 1
                name_to_new_id[name] = next_id
                used_ids.add(next_id)
                next_id += 1

        # Step 5: Build old_id → new_id mapping
        old_to_new = {old_id: name_to_new_id[name] for old_id, name in res}

        # Step 6: Temporarily shift IDs to avoid collisions
        update_statement_ftr_type = f"""
            UPDATE FeatureType
            SET id = ?
            WHERE id = ?
            """
        update_statement_ftr = f"""
            UPDATE Feature
            SET ftrType = ?
            WHERE ftrType = ?
            """
        TEMP_OFFSET = 1000
        update_values = [(old_id + TEMP_OFFSET, old_id) for old_id in old_to_new]
        self._update_many(update_statement_ftr_type, update_values) # Shift Feature Type Table
        self._update_many(update_statement_ftr, update_values) # Feature Table

        # Step 7: Assign Final IDs from Temp Status
        update_values = [(new_id, old_id + TEMP_OFFSET) for old_id, new_id in old_to_new.items()] 
        self._update_many(update_statement_ftr_type, update_values) # Shift Feature Type Table
        self._update_many(update_statement_ftr, update_values) # Feature Table
    