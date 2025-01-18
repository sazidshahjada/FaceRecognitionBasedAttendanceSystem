# Import libraries
import os
import torch
import cv2
import numpy as np
from facenet_pytorch import InceptionResnetV1
from scripts.detect_face import detect_faces_mtcnn
from tqdm import tqdm


model = InceptionResnetV1(pretrained='vggface2').eval()

def process_and_save_embeddings(raw_images_dir, output_path):
    embeddings = []
    labels = []

    for filename in tqdm(os.listdir(raw_images_dir), desc="Processing images"):
        file_path = os.path.join(raw_images_dir, filename)
        if not (filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg')):
            continue  # Skip non-image files

        # Load the image
        image = cv2.imread(file_path)
        if image is None:
            print(f"Skipping {filename}, unable to read.")
            continue
        
        # Detect faces
        faces, _ = detect_faces_mtcnn(image)
        
        # If no face is detected, skip the image
        if len(faces) == 0:
            print(f"No face detected in {filename}, skipping.")
            continue
        
        # Assuming the first detected face is the desired one
        face = faces[0]
        
        # Preprocess face for embedding model
        face = cv2.resize(face, (160, 160))  # Resize to 160x160
        face = torch.tensor(face).permute(2, 0, 1).float() / 255.0  # Normalize and rearrange to (C, H, W)
        face = face.unsqueeze(0)  # Add batch dimension
        
        # Generate embedding
        with torch.no_grad():
            embedding = model(face).squeeze(0).numpy()
        
        label = os.path.splitext(filename)[0]
        # Append results
        embeddings.append(embedding)
        labels.append(filename)  # Label with the image filename
    
    # Save embeddings and labels
    np.savez(output_path, embeddings=np.array(embeddings), labels=np.array(labels))
    print(f"Embeddings saved to {output_path}")
