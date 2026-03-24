from ultralytics import YOLO
import torch

# Load a model
model = YOLO("yolo26n.pt") # build a new model from scratch
#model = YOLO("runs/detect/train/weights/best.pt")  # load a pretrained model

# Use the model
results = model.train(data="data/training_dataset/data.yaml", epochs=1, imgsz=640)  # train the model

#torch.save(model.state_dict(), 'runs/detect/640_weld_32_epoch/model_state_dict.pt')
