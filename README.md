# FaceRecognitionBasedAttendanceSystem
## Author - Md Shahjada Sajid

## Overview
This project is a Face Recognition Based Attendance System that uses facial recognition to mark attendance. It leverages the `facenet-pytorch` library for face detection and embedding generation, and `scikit-learn` for similarity measurement.

## Features
- Save all students' embeddings
- Start the attendance system
- Show attendance history

## Requirements
- Python 3.11
- Required Python packages (listed in `requirements.txt`)

## Installation
1. Clone the repository:
    ```sh
    git clone https://github.com/sazidshahjada/FaceRecognitionBasedAttendanceSystem.git
    cd FaceRecognitionBasedAttendanceSystem
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage
1. Run the main script to access the menu:
    ```sh
    ./run.sh
    ```

2. Choose an option from the menu:
    - `1`: Save All Students Embedding
    - `2`: Start Attendance System
    - `3`: Show Attendance History
    - `4`: Exit

## File Descriptions
- [run.sh](http://_vscodecontentref_/1): Main script to run the system.
- [storing_embedding.py](http://_vscodecontentref_/2): Script to save all students' embeddings.
- [face_attendance.py](http://_vscodecontentref_/3): Script to start the attendance system.
- [show_attendance_df.py](http://_vscodecontentref_/4): Script to show attendance history.
- [requirements.txt](http://_vscodecontentref_/5): List of required Python packages.
- [data](http://_vscodecontentref_/6): Directory containing attendance logs and embeddings.
- [scripts](http://_vscodecontentref_/7): Directory containing helper scripts for face detection and attendance system.

## License
This project is licensed under the MIT License. See the [LICENSE](http://_vscodecontentref_/8) file for more details.

## Acknowledgements
- [facenet-pytorch](https://github.com/timesler/facenet-pytorch)
- [scikit-learn](https://scikit-learn.org/)