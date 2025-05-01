import sys
import cv2
import os
import numpy as np
import face_recognition
from datetime import datetime, date
import geocoder
import logging
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QTextEdit, 
                            QInputDialog, QMessageBox, QDialog)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
import dlib
from scipy.spatial import distance as dist
import mysql.connector
from mysql.connector import Error

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'root@1234',
    'database': 'attendance_system'
}

# Create directories
if not os.path.exists('training_imgs'):
    os.makedirs('training_imgs')
if not os.path.exists('attendance'):
    os.makedirs('attendance')
if not os.path.exists('assets'):
    os.makedirs('assets')
    logging.debug("Created assets directory")

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    attendance_signal = pyqtSignal(str)
    blink_signal = pyqtSignal(bool)
    error_signal = pyqtSignal(str)

    def __init__(self, known_faces, known_names):
        super().__init__()
        self.known_face_encodings = known_faces
        self.known_face_names = known_names
        self.running = True
        try:
            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        except Exception as e:
            logging.error(f"Failed to initialize dlib: {str(e)}")
            self.error_signal.emit(f"Failed to initialize dlib: {str(e)}")
            self.running = False
        self.lStart, self.lEnd = 42, 48
        self.rStart, self.rEnd = 36, 42

    def eye_aspect_ratio(self, eye):
        if len(eye) < 6:
            return float('inf')
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def is_blinking(self, frame, blink_thresh=0.25):  # Increased threshold for easier detection
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = self.detector(gray, 0)
            for rect in rects:
                shape = self.predictor(gray, rect)
                shape_np = np.array([[p.x, p.y] for p in shape.parts()])
                leftEye = shape_np[self.lStart:self.lEnd]
                rightEye = shape_np[self.rStart:self.rEnd]
                leftEAR = self.eye_aspect_ratio(leftEye)
                rightEAR = self.eye_aspect_ratio(rightEye)
                if leftEAR == float('inf') or rightEAR == float('inf'):
                    continue
                ear = (leftEAR + rightEAR) / 2.0
                logging.debug(f"EAR: {ear:.3f}, Threshold: {blink_thresh}")
                if ear < blink_thresh:
                    return True
            return False
        except Exception as e:
            logging.error(f"Error in is_blinking: {str(e)}")
            return False

    def run(self):
        logging.debug("Starting VideoThread")
        logging.debug(f"Known faces: {len(self.known_face_encodings)}")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logging.error("Failed to open webcam")
            self.error_signal.emit("Failed to open webcam")
            return

        blink_frames = 0
        BLINK_CONSEC_FRAMES = 2
        blink_detected = False

        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    logging.error("Failed to capture frame")
                    self.error_signal.emit("Failed to capture frame")
                    break

                try:
                    if self.is_blinking(frame):
                        blink_frames += 1
                        cv2.putText(frame, "Blink detected!", (30, 50), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        if blink_frames >= BLINK_CONSEC_FRAMES:
                            blink_detected = True
                            self.blink_signal.emit(True)
                    else:
                        blink_frames = 0
                        if not blink_detected:
                            cv2.putText(frame, "Please blink to verify", (30, 50), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    if blink_detected:
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        face_locations = face_recognition.face_locations(rgb_frame)
                        logging.debug(f"Detected {len(face_locations)} faces")
                        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

                        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                            logging.debug(f"Matches: {matches}")
                            name = "Unknown"
                            if True in matches:
                                first_match_index = matches.index(True)
                                name = self.known_face_names[first_match_index]
                                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                                cv2.putText(frame, name, (left, top-10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                                logging.debug(f"Emitting attendance signal for {name}")
                                self.attendance_signal.emit(name)
                            else:
                                logging.debug("No match found for detected face")

                    # Load background image
                    background = cv2.imread('assets/background.png')
                    if background is None:
                        background = np.ones((480, 640, 3), dtype=np.uint8) * 45
                        logging.debug("Using default background")

                    # Resize background to match frame
                    frame_height, frame_width = frame.shape[:2]
                    bg_resized = cv2.resize(background, (frame_width, frame_height))
                    # Blend the background with the frame
                    frame = cv2.addWeighted(frame, 0.9, bg_resized, 0.1, 0)

                    self.change_pixmap_signal.emit(frame)
                except Exception as e:
                    logging.error(f"Error processing frame: {str(e)}")
                    self.error_signal.emit(f"Error processing frame: {str(e)}")
                    break
        finally:
            cap.release()
            logging.debug("VideoThread terminated")

    def stop(self):
        self.running = False
        self.wait()

class AttendanceRecordsDialog(QDialog):
    def __init__(self, uid, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Attendance Records for UID: {uid}")
        self.setStyleSheet("""
            QDialog {
                background-color: #1a1a1a;
            }
            QLabel {
                color: white;
                font-size: 16px;
            }
            QTextEdit {
                background-color: #2d2d2d;
                color: white;
                border-radius: 5px;
                padding: 10px;
                font-size: 14px;
            }
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
                font-size: 14px;
                min-width: 100px;
                margin: 5px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #0D47A1;
            }
        """)
        self.setFixedSize(600, 400)

        layout = QVBoxLayout()
        title_label = QLabel(f"Attendance Records for UID: {uid}")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #2196F3; margin: 10px;")
        layout.addWidget(title_label, alignment=Qt.AlignmentFlag.AlignCenter)

        self.records_text = QTextEdit()
        self.records_text.setReadOnly(True)
        layout.addWidget(self.records_text)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn, alignment=Qt.AlignmentFlag.AlignCenter)

        self.setLayout(layout)
        self.load_records(uid)

    def load_records(self, uid):
        logging.debug(f"Loading attendance records for UID: {uid}")
        try:
            connection = mysql.connector.connect(**DB_CONFIG)
            if connection.is_connected():
                cursor = connection.cursor()
                cursor.execute("""
                    SELECT u.user_id, u.name, a.timestamp, a.latitude, a.longitude
                    FROM attendance a
                    JOIN users u ON a.user_id = u.user_id
                    WHERE u.user_id = %s
                    ORDER BY a.timestamp DESC
                """, (uid,))
                records = cursor.fetchall()
                if not records:
                    self.records_text.append(f"No attendance records found for UID: {uid}")
                    logging.debug(f"No records found for UID: {uid}")
                    return

                for user_id, name, timestamp, latitude, longitude in records:
                    lat_str = f"{latitude:.6f}" if latitude is not None else "N/A"
                    lng_str = f"{longitude:.6f}" if longitude is not None else "N/A"
                    self.records_text.append(
                        f"ID: {user_id} ({name}) - {timestamp} - Lat: {lat_str}, Lng: {lng_str}"
                    )
                logging.debug(f"Loaded {len(records)} records for UID: {uid}")
        except Error as e:
            logging.error(f"Database error in load_records: {str(e)}")
            self.records_text.append(f"Error loading records: {str(e)}")
        finally:
            if 'connection' in locals() and connection.is_connected():
                cursor.close()
                connection.close()

class ModernAttendanceSystem(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Modern Attendance System")
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1a1a1a;
            }
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
                font-size: 14px;
                min-width: 200px;
                margin: 5px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #0D47A1;
            }
            QLabel {
                color: white;
                font-size: 16px;
            }
            QTextEdit {
                background-color: #2d2d2d;
                color: white;
                border-radius: 5px;
                padding: 10px;
                font-size: 14px;
            }
            QMessageBox {
                background-color: #2d2d2d;
                color: white;
            }
        """)

        # Initialize face recognition variables
        self.known_face_encodings = []
        self.known_face_names = []
        self.last_attendance = {}
        self.recent_attendance = []
        self.user_id_map = {}

        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        # Create left panel for video
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        self.image_label = QLabel()
        self.image_label.setFixedSize(640, 480)
        self.image_label.setStyleSheet("border: 2px solid #2196F3; border-radius: 10px;")
        left_layout.addWidget(self.image_label)
        layout.addWidget(left_panel)

        # Create right panel for controls and attendance
        right_panel = QWidget()
        right_panel.setFixedWidth(300)
        right_layout = QVBoxLayout(right_panel)

        # Title
        title_label = QLabel("Attendance System")
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #2196F3; margin: 20px 0;")
        right_layout.addWidget(title_label, alignment=Qt.AlignmentFlag.AlignCenter)

        # Add buttons
        register_btn = QPushButton("Register New Face")
        register_btn.clicked.connect(self.register_new_face)
        right_layout.addWidget(register_btn)

        start_btn = QPushButton("Start Attendance")
        start_btn.clicked.connect(self.start_attendance)
        right_layout.addWidget(start_btn)

        stop_btn = QPushButton("Stop Attendance")
        stop_btn.clicked.connect(self.stop_attendance)
        right_layout.addWidget(stop_btn)

        view_records_btn = QPushButton("View Attendance Records")
        view_records_btn.clicked.connect(self.view_attendance_records)
        right_layout.addWidget(view_records_btn)

        # Add attendance display
        attendance_label = QLabel("Recent Attendance")
        attendance_label.setStyleSheet("margin-top: 20px;")
        right_layout.addWidget(attendance_label)

        self.attendance_text = QTextEdit()
        self.attendance_text.setReadOnly(True)
        right_layout.addWidget(self.attendance_text)

        layout.addWidget(right_panel)

        # Initialize video thread
        self.thread = None
        self.load_known_faces()

    def load_known_faces(self):
        logging.debug("Loading known faces")
        try:
            connection = mysql.connector.connect(**DB_CONFIG)
            if connection.is_connected():
                cursor = connection.cursor()
                cursor.execute("SELECT user_id, name FROM users")
                users = cursor.fetchall()
                logging.debug(f"Found {len(users)} users in database")
                for user_id, name in users:
                    for image_file in os.listdir('training_imgs'):
                        if image_file.startswith(f"{name}_") and image_file.endswith(".jpg"):
                            image_path = os.path.join('training_imgs', image_file)
                            image = face_recognition.load_image_file(image_path)
                            encodings = face_recognition.face_encodings(image)
                            if encodings:
                                encoding = encodings[0]
                                self.known_face_encodings.append(encoding)
                                self.known_face_names.append(name)
                                self.user_id_map[name] = user_id
                                logging.debug(f"Loaded face for {name} (ID: {user_id})")
                            else:
                                logging.warning(f"No face encodings found in {image_file}")
                logging.debug(f"Loaded {len(self.known_face_encodings)} face encodings")
        except Error as e:
            logging.error(f"Database error in load_known_faces: {str(e)}")
            QMessageBox.critical(self, "Database Error", f"Failed to load faces: {e}")
        finally:
            if 'connection' in locals() and connection.is_connected():
                cursor.close()
                connection.close()

    def register_new_face(self):
        logging.debug("Starting face registration")
        name, ok = QInputDialog.getText(self, "Register New Face", "Enter name:")
        if ok and name:
            user_id, ok = QInputDialog.getText(self, "Register New Face", "Enter unique User ID (e.g., ASK001):")
            if ok and user_id:
                try:
                    connection = mysql.connector.connect(**DB_CONFIG)
                    if connection.is_connected():
                        cursor = connection.cursor()
                        cursor.execute("SELECT id FROM users WHERE user_id = %s", (user_id,))
                        if cursor.fetchone():
                            QMessageBox.critical(self, "Error", f"User ID {user_id} already exists!")
                            return
                        cursor.execute("INSERT INTO users (user_id, name) VALUES (%s, %s)", 
                                    (user_id, name))
                        connection.commit()
                        
                        cap = cv2.VideoCapture(0)
                        if not cap.isOpened():
                            QMessageBox.critical(self, "Error", "Failed to open webcam")
                            logging.error("Failed to open webcam for registration")
                            return

                        try:
                            for i in range(5):
                                ret, frame = cap.read()
                                if not ret:
                                    QMessageBox.critical(self, "Error", "Failed to capture frame")
                                    logging.error("Failed to capture frame during registration")
                                    break
                                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                face_locations = face_recognition.face_locations(rgb_frame)
                                if face_locations:
                                    encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                                    if encodings:
                                        self.known_face_encodings.append(encodings[0])
                                        self.known_face_names.append(name)
                                        self.user_id_map[name] = user_id
                                        cv2.imwrite(f'training_imgs/{name}_{i}.jpg', frame)
                                        logging.debug(f"Saved image {i+1} for {name}")
                                    else:
                                        logging.warning("No face encodings found during registration")
                                else:
                                    logging.warning("No faces detected during registration")
                            QMessageBox.information(self, "Success", f"Registration complete for {name}")
                        finally:
                            cap.release()
                            logging.debug("Camera released after registration")
                except Error as e:
                    logging.error(f"Database error in register_new_face: {str(e)}")
                    QMessageBox.critical(self, "Database Error", f"Failed to register user: {e}")
                finally:
                    if 'connection' in locals() and connection.is_connected():
                        cursor.close()
                        connection.close()

    def view_attendance_records(self):
        logging.debug("Opening view attendance records dialog")
        uid, ok = QInputDialog.getText(self, "View Attendance Records", "Enter User ID (e.g., ASK001):")
        if ok and uid:
            dialog = AttendanceRecordsDialog(uid, self)
            dialog.exec()

    def start_attendance(self):
        logging.debug("Starting attendance")
        if self.thread is None:
            self.thread = VideoThread(self.known_face_encodings, self.known_face_names)
            self.thread.change_pixmap_signal.connect(self.update_image)
            self.thread.attendance_signal.connect(self.mark_attendance)
            self.thread.error_signal.connect(self.handle_thread_error)
            self.thread.blink_signal.connect(self.handle_blink)
            self.thread.start()
        else:
            logging.warning("VideoThread already running")

    def stop_attendance(self):
        logging.debug("Stopping attendance")
        if self.thread is not None:
            self.thread.stop()
            self.thread = None
            self.image_label.clear()
            logging.debug("VideoThread stopped")

    def handle_thread_error(self, error_msg):
        logging.error(f"Thread error: {error_msg}")
        QMessageBox.critical(self, "Error", error_msg)
        self.stop_attendance()

    def handle_blink(self, detected):
        logging.debug(f"Blink detected: {detected}")

    def update_image(self, frame):
        try:
            qt_img = self.convert_cv_qt(frame)
            self.image_label.setPixmap(qt_img)
        except Exception as e:
            logging.error(f"Error updating image: {str(e)}")
            self.handle_thread_error(f"Error updating image: {str(e)}")

    def convert_cv_qt(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        scaled = convert_to_qt_format.scaled(640, 480, Qt.AspectRatioMode.KeepAspectRatio)
        return QPixmap.fromImage(scaled)

    def mark_attendance(self, name):
        logging.debug(f"Attempting to mark attendance for {name}")
        try:
            connection = mysql.connector.connect(**DB_CONFIG)
            if connection.is_connected():
                cursor = connection.cursor()
                user_id = self.user_id_map.get(name)
                if not user_id:
                    logging.warning(f"No user ID found for {name} in user_id_map")
                    return

                today = date.today().isoformat()
                if self.last_attendance.get(name) == today:
                    logging.debug(f"Attendance already marked for {name} today")
                    return

                self.last_attendance[name] = today
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                logging.debug(f"Generated timestamp: {timestamp}")
                
                try:
                    location = geocoder.ip("me")
                    lat, lng = location.latlng if location else (0.0, 0.0)
                    logging.debug(f"Location: Lat={lat}, Lng={lng}")
                except Exception as e:
                    lat, lng = 0.0, 0.0
                    logging.warning(f"Failed to fetch location: {str(e)}, using default (0, 0)")

                logging.debug(f"Inserting attendance: user_id={user_id}, timestamp={timestamp}, lat={lat}, lng={lng}")
                cursor.execute(
                    "INSERT INTO attendance (user_id, timestamp, latitude, longitude) VALUES (%s, %s, %s, %s)",
                    (user_id, timestamp, lat, lng)
                )
                connection.commit()
                logging.debug("Database insertion committed")

                attendance_text = f"{name} (ID: {user_id}) - {timestamp}\n"
                self.attendance_text.append(attendance_text)
                self.recent_attendance.append(attendance_text)
                logging.debug(f"Attendance marked for {name} (ID: {user_id})")
        except Error as e:
            logging.error(f"Database error in mark_attendance: {str(e)}")
            QMessageBox.critical(self, "Database Error", f"Failed to mark attendance: {e}")
        except Exception as e:
            logging.error(f"Unexpected error in mark_attendance: {str(e)}")
            QMessageBox.critical(self, "Error", f"Unexpected error: {e}")
        finally:
            if 'connection' in locals() and connection.is_connected():
                cursor.close()
                connection.close()
                logging.debug("Database connection closed")

    def closeEvent(self, event):
        logging.debug("Closing application")
        if self.thread is not None:
            self.thread.stop()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ModernAttendanceSystem()
    window.setGeometry(100, 100, 1200, 600)
    window.show()
    try:
        sys.exit(app.exec())
    except Exception as e:
        logging.error(f"Application crashed: {str(e)}")
