import warnings
import pandas as pd

warnings.filterwarnings("ignore")

from scripts.attendance_system import show_attendance

# Define path to attendance CSV
csv_file = './data/attendance_logs/attendance.csv'

# Show attendance
show_attendance(csv_file)
# Expected output:
# +-----------------+---------------------+
# | Student ID      | Time                |
# +-----------------+---------------------+
# | 123456          | 2021-08-01 12:00:00 |
# | 234567          | 2021-08-01 12:01:00 |
# | 345678          | 2021-08-01 12:02:00 |
