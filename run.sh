#!/bin/bash
echo

echo "Welcome to Face Recognition Attendance System"
while true; do
    echo
    echo "Please choose an option:"
    echo "1. Save All Students Embedding"
    echo "2. Start Attendance System"
    echo "3. Show Attendance History"
    echo "4. Exit"

    read -p "Enter your choice: " choice

    echo

    case $choice in
        1)
            python3 storing_embedding.py
            ;;
        2)
            python3 face_attendance.py
            ;;
        3)
            python3 show_attendance_df.py
            ;;
        4)
            echo "Exiting..."
            exit 0
            ;;
        *)
            echo "ValueError: Invalid choice. Please enter a number between 1 and 4."
            ;;
    esac
done