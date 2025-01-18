# Import libraries
import os
import torch
import cv2
import numpy as np
from facenet_pytorch import MTCNN


# Initialize MTCNN
mtcnn = MTCNN(keep_all=True)

# Function to detect faces using MTCNN
def detect_faces_mtcnn(image):
    # Convert BGR to RGB as MTCNN expects RGB format
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Detect faces and bounding boxes
    boxes, _ = mtcnn.detect(image_rgb)
    
    if boxes is None:  # No faces detected
        return [], []

    # Crop faces based on bounding boxes
    faces = []
    for box in boxes:
        x1, y1, x2, y2 = [int(coord) for coord in box]
        face = image[y1:y2, x1:x2]  # Crop face
        faces.append(face)
    
    return faces, boxes

if __name__ == "__main__":
   pass