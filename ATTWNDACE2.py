import cv2
import os
import numpy as np
import face_recognition
import tkinter as tk
from tkinter import simpledialog,messagebox
from datetime import datetime
import geocoder

#intialize the cmaera
cap =cv2.VideoCapture(0) #use to capture the video input and 0 indicates the deafault camera

# creating directories to store data to train and attendance
if not os.path.exists('training_imgs'):
    os.makedirs('training_imgs')
if not os.path.exists('attendance'):
    os.makedirs('attendance')

#creating an empty list to store the data, face_encodings and known_face

known_face_encodings =[]
known_face_names = []

# creating a function to register th face using cv2 for input
def register_face(name):
    print(f"registering {name}pls look at camera")
    for i in range(20): #it will take 20 pics for data
        ret,frame=cap.read()
        if not ret:
            print('failed to capture face')
            break

        #converting img to rgb
        rgb_frame= cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        # find face in rgb imgs
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame,face_locations)
        # if faces found , save them
        if face_encodings:
            for encodings in face_encodings:
                known_face_encodings.append(encodings)
                known_face_names.append(name)
                print(f"image{i+1}saved for {name}")
        # captured img save
        cv2.imwrite(f'training_imgs/{name}_{i}.jpg', frame)
    messagebox.showinfo("Success", f"Registration for {name} completed.")

# function to train the model using data taken
def train_model():
    print ("trainig the model")
    # load the image and extract encoding
    for image_file in os.listdir('training_imgs'):
        if image_file.endswith(".jpg"):
            image_path = os.path.join('training_imgs',image_file)
            image = cv2.imread(image_path)
            rgb_image =cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

            # finding faces in image
            face_locations= face_recognition.face_locations(rgb_image)
            face_encodings= face_recognition.face_encodings(rgb_image,face_locations)
            # add encodings to list
            for encodings in face_encodings:
                known_face_encodings.append(encodings)
                name =image_file.split('_')[0] # use the prefix form filename
                known_face_names.append(name)
# func to mark the attendance
def mark_attendance(name):
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d %H:%M:%S")
    location = geocoder.ip("me")
    current_loc =(location.latlng)
    # add the attendace in a file
    with open('attendance/attendance_log.csv','a') as f:
        f.write(f"{name},{current_time},{current_loc}\n")
    print(f"attendace marked for {name} at {current_time} at {current_loc}")

# main loop to recofnize face
def take_attendance():
    while True:
        ret,frame = cap.read()
        if not ret:
            print("failed to capture img")
            break
        rgb_frame =cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        # find face in frame
        face_locations =face_recognition.face_locations(rgb_frame)
        face_encodings =face_recognition.face_encodings(rgb_frame,face_locations)

        for(top,right,bottom,left),face_encodings in zip(face_locations,face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings ,face_encodings)
            name = "unknown"

            if True in matches:
                first_match_index =matches.index(True)
                name = known_face_names[first_match_index]

                #draw a box around the face,add name
                cv2.rectangle(frame,(left,top),(right,bottom),(0,255,0),2)
                cv2.putText(frame,name,(left,top-10),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)

                # MARK ATTENDANCE A RECOGNOZED FACE IS DETECTED
                mark_attendance(name)
                # display the imge with rslt
        cv2.imshow("attendace systm",frame)
        # press q to quit
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
