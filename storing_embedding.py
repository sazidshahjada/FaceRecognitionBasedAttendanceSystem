import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from facenet_pytorch import MTCNN, InceptionResnetV1
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="facenet_pytorch")

import sys
sys.path.append('/home/sajid/Work/FaceRecognitionBasedAttendanceSystem')

from scripts.detect_face import detect_faces_mtcnn
from scripts.bulk_embedding import process_and_save_embeddings

# Load models
mtcnn = MTCNN(keep_all=True)
model = InceptionResnetV1(pretrained='vggface2').eval()

# Define paths
raw_images_dir = './data/raw_images'
output_path = './data/embeddings/embeddings.npz'

# Process and save embeddings
process_and_save_embeddings(raw_images_dir, output_path)