import cv2
import torch
import numpy as np
import mediapipe as mp
from ultralytics import YOLO

# Load YOLOv8 model for object detection
model = YOLO("yolov8n.pt")  # Use "yolov8s.pt" for better accuracy

# Load Face Recognizer Model
try:
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
except AttributeError:
    print("Error: OpenCV face module not found. Install opencv-contrib-python.")
    exit()
face_recognizer.read("models/trained_face_model.yml")

# Load Face Labels
labels = {}
with open("models/labels.txt", "r") as f:
    for line in f:
        label_id, name = line.strip().split(",")
        labels[int(label_id)] = name

# Load OpenCV Face Detector (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Initialize Pose Detector (MediaPipe)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Open Webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to exit.")

# Store Face Tracking Information
face_history = {}  # {ID -> Name}
next_id = 1  # Unique ID for unknown persons

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Step 1: Face Detection & Recognition
    faces_detected = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in faces_detected:
        face_roi = gray[y:y+h, x:x+w]

        # Recognize Face
        label_id, confidence = face_recognizer.predict(face_roi)
        name = labels.get(label_id, "Unknown") if confidence < 80 else "Unknown"

        # Assign Unique ID to Unknown Persons
        found_id = next((prev_id for prev_id, prev_name in face_history.items() if prev_name == name), None)
        if found_id is None:
            found_id = next_id
            face_history[found_id] = name
            next_id += 1

        # Draw Bounding Box & Label for Face
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, f"{name} (ID: {found_id})", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Step 2: Object Detection with YOLO
    results = model(frame)

    for result in results:
        for obj in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = obj
            label = model.names[int(class_id)]

            # Draw Bounding Box & Label for Object
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(frame, f"{label} ({score:.2f})", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Step 3: Pose Detection for Unknown Persons
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_pose = pose.process(rgb_frame)

    if results_pose.pose_landmarks:
        for landmark in results_pose.pose_landmarks.landmark:
            h, w, _ = frame.shape
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)

    # Display Output
    cv2.imshow("Face, Object & Pose Tracking", frame)

    # Press 'q' to Exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release Resources
cap.release()
cv2.destroyAllWindows()
