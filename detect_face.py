import cv2
import numpy as np

model_path = "models/trained_face_model.yml"
labels_path = "models/labels.txt"

# Load trained model
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read(model_path)

# Load labels
labels = {}
with open(labels_path, "r") as f:
    for line in f:
        label_id, name = line.strip().split(",")
        labels[int(label_id)] = name

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_detected = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in faces_detected:
        face_roi = gray[y:y+h, x:x+w]

        # Recognize the face
        label_id, confidence = face_recognizer.predict(face_roi)

        if confidence < 80:  # Adjust threshold as needed
            name = labels.get(label_id, "Unknown")
        else:
            name = "Unknown"

        # Draw rectangle and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{name} ({confidence:.2f})", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
