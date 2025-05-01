# Modern Attendance System

## Overview

The Modern Attendance System is a Python-based application that automates attendance marking using face recognition and liveness detection (blink detection). It leverages computer vision libraries like OpenCV and `face_recognition`, along with a PyQt6 GUI for user interaction. The system stores attendance records in a MySQL database and provides features to register new users, mark attendance, and view historical records for specific users. It is designed for educational or organizational environments to streamline attendance tracking with a secure and user-friendly interface.

## Features

- **Face Recognition**: Identifies users using facial recognition powered by the `face_recognition` library.
- **Blink Detection**: Ensures liveness by requiring users to blink, preventing spoofing with static images.
- **MySQL Integration**: Stores user data and attendance records in a MySQL database.
- **User Registration**: Allows registering new users by capturing face images and storing them for recognition.
- **Attendance Marking**: Automatically marks attendance with timestamp and geolocation (latitude, longitude).
- **Attendance Records**: View all attendance records for a specific user ID in a separate dialog window.
- **Modern GUI**: Built with PyQt6, featuring a dark theme with intuitive controls.
- **Error Handling and Logging**: Includes robust error handling and detailed logging for debugging.

## Screenshots

### Main Interface (Before Starting Attendance)
The main window displays a video feed area (initially blank), control buttons, and a recent attendance log.

![Main Interface Before Starting Attendance](screenshots/main_interface_before.png)
![image](https://github.com/user-attachments/assets/147e000a-ddb0-4f7f-80a3-e904f778a352)


### Marking Attendance
When attendance is started, the system detects blinks ("Blink detected!") and recognizes faces, marking attendance in real-time. The recognized user's name and ID are displayed on the video feed, and the attendance is logged in the "Recent Attendance" section.

![Marking Attendance](screenshots/marking_attendance.png)

![image](https://github.com/user-attachments/assets/0aa2ebca-8756-41b5-9436-93aee2adc9f3)

### Viewing Attendance Records
Users can view all attendance records for a specific UID (e.g., `ASHU001`). Records include the user ID, name, timestamp, and geolocation.

![Viewing Attendance Records](screenshots/attendance_records.png)
![image](https://github.com/user-attachments/assets/ccb18c8a-5f7b-4379-8432-3c312633ff13)

## Requirements

- **Python 3.8+**
- **MySQL Server** (e.g., MySQL Community Server)
- **Webcam** (for face capture and attendance marking)
- **Dependencies**:
  - `opencv-python`
  - `numpy`
  - `face_recognition`
  - `mysql-connector-python`
  - `geocoder`
  - `dlib`
  - `scipy`
  - `PyQt6`

## Setup Instructions

### 1. Clone the Repository
Clone this project to your local machine:
```bash
git clone <repository-url>
cd modern-attendance-system
