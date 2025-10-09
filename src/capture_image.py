import cv2
import os

# Create dataset directory if not exists
dataset_path = "dataset_faces"
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

# Ask for the person's name (used as folder name)
person_name = input("Enter the person's name: ")
person_path = os.path.join(dataset_path, person_name)

# Create person's directory if not exists
if not os.path.exists(person_path):
    os.makedirs(person_path)

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print(f"Capturing images for {person_name}. Press 'q' to stop.")

img_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Display the frame
    cv2.imshow("Face Capture", frame)

    # Save the frame
    img_filename = os.path.join(person_path, f"image_{img_count}.jpg")
    cv2.imwrite(img_filename, frame)
    img_count += 1

    # Press 'q' to quit
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
print(f"✅ {img_count} images saved for {person_name} in {person_path}")
