import cv2
import os
import numpy as np
import face_recognition
from datetime import datetime
import tkinter as tk
from tkinter import simpledialog, messagebox

# Initialize the camera
cap = cv2.VideoCapture(0)

# Create directories for storing images and encoded faces
if not os.path.exists('training_images'):
    os.makedirs('training_images')
if not os.path.exists('attendance'):
    os.makedirs('attendance')

# List for storing face encodings and corresponding names
known_face_encodings = []
known_face_names = []

# Function to register a face (Capture image and save it)
def register_face(name):
    print(f"Registering {name}, please look at the camera...")
    for i in range(20):  # Capture 20 images for training
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Convert image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Find faces in the frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        # If faces are found, save them
        if face_encodings:
            for encoding in face_encodings:
                # Save the face encoding and name
                known_face_encodings.append(encoding)
                known_face_names.append(name)
                print(f"Image {i+1} saved for {name}")
        
        # Save the captured image
        cv2.imwrite(f'training_images/{name}_{i}.jpg', frame)
    
    messagebox.showinfo("Registration", f"Registration for {name} complete!")

# Function to train the face recognition model (encodes faces from the stored images)
def train_model():
    print("Training the model...")
    # Load images and extract face encodings
    for image_file in os.listdir('training_images'):
        if image_file.endswith(".jpg"):
            image_path = os.path.join('training_images', image_file)
            image = cv2.imread(image_path)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Find all faces in the image
            face_locations = face_recognition.face_locations(rgb_image)
            face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

            # Add the encodings to the list
            for encoding in face_encodings:
                known_face_encodings.append(encoding)
                name = image_file.split('_')[0]  # Use the filename prefix as the name
                known_face_names.append(name)

# Function to mark attendance based on the recognized face
def mark_attendance(name):
    # Get the current date and time
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d %H:%M:%S")

    # Log the attendance to a file
    with open('attendance/attendance_log.csv', 'a') as f:
        f.write(f"{name},{current_time}\n")
    print(f"Attendance marked for {name} at {current_time}")

# Main loop to recognize faces and take attendance
def take_attendance():
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Find faces in the frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

                # Draw a box around the face and label it with the name
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                # Mark attendance when a recognized face is detected
                mark_attendance(name)
        
        # Display the image with the face recognition results
        cv2.imshow("Attendance System", frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# GUI interface using Tkinter
class AttendanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition Attendance System")
        
        # Create buttons
        self.register_button = tk.Button(root, text="Register New Face", width=30, height=2, command=self.register_new_face)
        self.register_button.pack(pady=10)

        self.start_button = tk.Button(root, text="Start Attendance", width=30, height=2, command=self.start_attendance)
        self.start_button.pack(pady=10)

        self.quit_button = tk.Button(root, text="Exit", width=30, height=2, command=self.exit_app)
        self.quit_button.pack(pady=10)
    
    def register_new_face(self):
        # Get name input
        name = simpledialog.askstring("Input", "Enter the name of the person:")
        if name:
            register_face(name)
    
    def start_attendance(self):
        # Train the model
        train_model()
        # Start attendance system
        take_attendance()
    
    def exit_app(self):
        cap.release()
        cv2.destroyAllWindows()
        self.root.quit()

# Run the Tkinter GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = AttendanceApp(root)
    root.mainloop()
