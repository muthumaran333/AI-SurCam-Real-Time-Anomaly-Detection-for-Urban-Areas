import cv2
import torch
import numpy as np
import mediapipe as mp
import smtplib
import ssl
import os
import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

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

# Load OpenCV Face Detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Initialize Pose Detector
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()


# Email Notification Function
def send_email_alert(known_name, unknown_id, frame):
    sender_email = "finalyearproject2812@gmail.com"
    receiver_email = "finalyearproject2812@gmail.com"
    password = "lghkfkivfgwmbfml"

    subject = "ALERT: Unknown Person Detected with Known Person"
    body = f"An unknown person (ID: {unknown_id}) was detected along with {known_name}. Please check the surveillance footage immediately."

    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    image_path = "detection_alert.jpg"
    cv2.imwrite(image_path, frame)

    with open(image_path, "rb") as img_file:
        img_attachment = MIMEImage(img_file.read(), name=os.path.basename(image_path))
        msg.attach(img_attachment)

    try:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
        print("✅ ALERT EMAIL WITH IMAGE SENT!")
    except Exception as e:
        print(f"❌ Error sending email: {e}")
    os.remove(image_path)


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

face_history = {}
unknown_timers = {}
notified_users = set()
next_id = 1

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_detected = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    detected_known = False

    for (x, y, w, h) in faces_detected:
        face_roi = gray[y:y + h, x:x + w]
        label_id, confidence = face_recognizer.predict(face_roi)
        name = labels.get(label_id, "Unknown") if confidence < 80 else "Unknown"

        found_id = next((pid for pid, pname in face_history.items() if pname == name), None)
        if found_id is None:
            found_id = next_id
            face_history[found_id] = name
            next_id += 1

        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{name} (ID: {found_id})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if name != "Unknown":
            detected_known = True
        else:
            if found_id not in unknown_timers:
                unknown_timers[found_id] = time.time()

            elapsed_time = time.time() - unknown_timers[found_id]

            # Send notification after 30 seconds of unknown presence
            if elapsed_time > 30 and found_id not in notified_users:
                known_name = next((pname for pid, pname in face_history.items() if pname != "Unknown"),
                                  "No Known Person")

                # Print reason for notification
                print(
                    f"🚨 ALERT: Unknown person (ID: {found_id}) detected along with {known_name}. Sending email notification...")

                send_email_alert(known_name, found_id, frame)
                notified_users.add(found_id)

    # Object Detection using YOLOv8
    results = model(frame)
    for result in results:
        for obj in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = obj
            label = model.names[int(class_id)]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(frame, f"{label} ({score:.2f})", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Pose Detection using MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_pose = pose.process(rgb_frame)
    if results_pose.pose_landmarks:
        for landmark in results_pose.pose_landmarks.landmark:
            h, w, _ = frame.shape
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)

    cv2.imshow("Face, Object & Pose Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
