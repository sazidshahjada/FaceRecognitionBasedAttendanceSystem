import os
import cv2
import csv
import torch
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from facenet_pytorch import MTCNN, InceptionResnetV1

# Initialize MTCNN for face detection
mtcnn = MTCNN(keep_all=True)

# Load the InceptionResnetV1 model for embedding generation
model = InceptionResnetV1(pretrained="vggface2").eval()

# Define the function to get face embeddings
def get_face_embedding(face, model):
    face = face / 255.0  # Normalize pixel values to [0, 1]
    face = np.transpose(face, (2, 0, 1))  # Change shape to [C, H, W]
    face = np.expand_dims(face, axis=0)  # Add batch dimension
    face_tensor = torch.tensor(face, dtype=torch.float32)
    embedding = model(face_tensor).detach().numpy()
    return embedding.squeeze()

# Updated attendance function
def update_attendance_csv(embeddings_file, csv_file, threshold=0.7):
    # Load embeddings and labels
    data = np.load(embeddings_file)
    saved_embeddings = data["embeddings"]
    labels = [os.path.splitext(label)[0] for label in data["labels"]]  # Remove file format

    # Initialize camera
    cap = cv2.VideoCapture(0)
    recognized_students = set()
    recognition_counters = {label: 0 for label in labels}  # Counter for each label

    print("Press 'q' to stop attendance recording.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image.")
            break

        # Detect faces
        boxes, _ = mtcnn.detect(frame)
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = [int(coord) for coord in box]

                # Ensure bounding box coordinates are within bounds
                h, w, _ = frame.shape
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                # Extract and preprocess face
                face = frame[y1:y2, x1:x2]
                if face.size == 0:
                    continue

                face = cv2.resize(face, (160, 160))
                face_embedding = get_face_embedding(face, model)

                # Compare embeddings
                similarities = cosine_similarity([face_embedding], saved_embeddings)
                max_similarity_index = np.argmax(similarities)
                max_similarity_score = similarities[0, max_similarity_index]

                if max_similarity_score >= threshold:
                    label = labels[max_similarity_index]
                    recognition_counters[label] += 1  # Increment counter for the label
                else:
                    label = "Unknown"

                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Reset counters for labels not detected in the current frame
        for label in recognition_counters:
            if label not in [labels[np.argmax(cosine_similarity([face_embedding], saved_embeddings))]]:
                recognition_counters[label] = 0

        # Add to recognized_students when counter reaches 20
        for label, count in recognition_counters.items():
            if count >= 20 and label not in recognized_students:
                recognized_students.add(label)

        # Display the frame
        cv2.imshow("Face Recognition", frame)

        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Update the attendance CSV
    date = str(np.datetime64('today', 'D'))
    if not recognized_students:
        print("No students recognized.")
        return

    try:
        # Read existing CSV or create a new one
        if not os.path.exists(csv_file):
            with open(csv_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Name", date])

        with open(csv_file, "r") as f:
            reader = csv.reader(f)
            headers = next(reader)
            attendance_data = {row[0]: row[1:] for row in reader}

        # Check if the date already exists
        if date not in headers:
            headers.append(date)
            for student in attendance_data:
                attendance_data[student].append("0")

        # Mark attendance only for recognized students
        date_index = headers.index(date)
        for student in recognized_students:
            if student not in attendance_data:
                attendance_data[student] = ["0"] * (len(headers) - 1)
            attendance_data[student][date_index - 1] = "1"

        # Ensure all unrecognized students remain unmarked
        for student in attendance_data:
            if student not in recognized_students and len(attendance_data[student]) < len(headers) - 1:
                attendance_data[student].append("0")

        # Write updated attendance data
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for student, records in attendance_data.items():
                writer.writerow([student] + records)

        print(f"Attendance updated successfully for {date}.")
    except Exception as e:
        print(f"Error updating attendance CSV: {e}")

# Function to show attendance from CSV
def show_attendance(csv_file):
    try:
        df = pd.read_csv(csv_file)
        print(df)
    except FileNotFoundError:
        print("Attendance CSV file not found.")