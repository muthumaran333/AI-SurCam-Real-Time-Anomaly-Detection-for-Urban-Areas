import cv2
import os
import numpy as np

dataset_path = "dataset_faces"
model_path = "models"

# Ensure models directory exists
if not os.path.exists(model_path):
    os.makedirs(model_path)

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

faces = []
labels = []
label_map = {}

# Read dataset
label_id = 0
for person in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person)

    if os.path.isdir(person_path):
        label_map[label_id] = person  # Store mapping for later use

        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue

            # Detect face in the image
            faces_detected = face_detector.detectMultiScale(img, scaleFactor=1.2, minNeighbors=5)
            for (x, y, w, h) in faces_detected:
                faces.append(img[y:y+h, x:x+w])
                labels.append(label_id)

        label_id += 1

# Train the model
if len(faces) > 0:
    face_recognizer.train(faces, np.array(labels))
    face_recognizer.save(os.path.join(model_path, "trained_face_model.yml"))

    # Save labels
    with open(os.path.join(model_path, "labels.txt"), "w") as f:
        for label_id, person in label_map.items():
            f.write(f"{label_id},{person}\n")

    print("✅ Face model trained and saved!")
else:
    print("⚠️ No faces detected in dataset!")

