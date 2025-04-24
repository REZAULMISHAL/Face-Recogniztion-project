import cv2
import face_recognition
import os
import time
from datetime import datetime, timedelta

# Load pre-registered known faces and names
known_faces = []
known_names = []

known_faces_dir = "faces"

for filename in os.listdir(known_faces_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        known_names.append(os.path.splitext(filename)[0])
        face_image = face_recognition.load_image_file(os.path.join(known_faces_dir, filename))
        face_encoding = face_recognition.face_encodings(face_image)[0]  # Assuming one face per image
        known_faces.append(face_encoding)

# Initialize webcam
video_capture = cv2.VideoCapture(0)

# Initialize a dictionary to track attendance times
attendance_records = {}

while True:
    ret, frame = video_capture.read()

    # Find all face locations and encodings in the current frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
        # Compare the current face encoding with the known face encodings
        matches = face_recognition.compare_faces(known_faces, face_encoding)
        name = "Unknown"

        # Check if there's a match
        if True in matches:
            matched_index = matches.index(True)
            name = known_names[matched_index]


        # Draw a box around the face and display the name
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

    # Display the resulting frame
    cv2.imshow("we are watching you", frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close all windows
video_capture.release()
cv2.destroyAllWindows()
