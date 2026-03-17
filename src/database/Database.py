import os
import math
import sqlite3

from src.models.detection import Detection

class DB:
    """
    Class Data:
        filepath - filepath of the database
        name - database file name
        conn - sqlite database connection

    Methods:
        select - execute sql select statement and return values
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
    

class RVHR_DB(DB):
    """
    Database class for RVHR. Table names are "Image, "Feature, "FeatureType"
    """

    def get_img_id(self, img_name: str):
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
    
    def get_img_name(self, img_id: int):
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
    
    def get_labelled_img_ids(self, ftr_types):
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

        image_name = self.get_img_name(img_id)
        annotations = []
        for row in res:
            label, x_min, y_min, x_max, y_max = row
            annotation = Detection(
                label = label,
                bbox = [x_min, y_min, x_max, y_max]
            )
            annotations.append(annotation)
        return annotations
    
