import os
import csv
import cv2
import warnings
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from facenet_pytorch import MTCNN, InceptionResnetV1

warnings.filterwarnings("ignore")


from scripts.detect_face import detect_faces_mtcnn
from scripts.attendance_system import get_face_embedding, update_attendance_csv

# Load models
mtcnn = MTCNN(keep_all=True)
model = InceptionResnetV1(pretrained='vggface2').eval()

if __name__ == '__main__':
    # Define paths
    embeddings_file = './data/embeddings/embeddings.npz'
    csv_file = './data/attendance_logs/attendance.csv'

    # Ensure directories exist
    os.makedirs(os.path.dirname(embeddings_file), exist_ok=True)
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)

    # Update attendance CSV
    update_attendance_csv(embeddings_file, csv_file, threshold=0.7)
    # Expected output: Press 'q' to stop attendance recording.
    # Note: The output may vary depending on the webcam feed and recognized faces.