# model_live_check.py
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import os

torch.cuda.empty_cache()
# Use relative path to the models directory
model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'best.pt')
model = YOLO(model_path)

def detect_and_segment(image, target_class_id=0):
    results = model.predict(source=image, show=False, save=False, stream=False)

    detections = []
    for result in results:
        if result.masks is not None:
            for i, cls_id in enumerate(result.boxes.cls.cpu().numpy()):
                # For single class model (bolt_assem), accept any detection
                # or you can check for specific class name if needed
                mask = result.masks.data[i].cpu().numpy()
                box = result.boxes.xyxy[i].cpu().numpy()
                detections.append({
                    'mask': mask,
                    'bbox': box,
                    'class_id': int(cls_id)
                })
    return detections
