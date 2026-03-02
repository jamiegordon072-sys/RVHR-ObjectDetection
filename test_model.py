from PIL import Image
from ultralytics import YOLO
import cv2
import os
import torch

# Load the YOLO model
model = YOLO("runs/detect/new dataset/weld_corrugation/weights/best.pt")

# Perform analysis on each file in the folder and save results
dir_path = "data/model_testing/training_data_test"
for path in os.listdir(dir_path):
    if os.path.isfile(os.path.join(dir_path, path)):
        img_path = dir_path + "/" + path
        results = model(img_path)
        for r in results:
            im_array = r.plot()
            im = Image.fromarray(im_array[..., ::-1])
            im.save(dir_path + "/results/" + path)