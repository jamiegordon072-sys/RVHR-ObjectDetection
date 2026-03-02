from ultralytics import YOLO
import torch

# Load a model
model = YOLO('runs/detect/train/weights/last.pt')  # load a partially trained model

# Resume training
results = model.train(resume=True)

torch.save(model.state_dict(), 'model_state_dict.pt')