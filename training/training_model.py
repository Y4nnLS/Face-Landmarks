import cv2
import os
import numpy as np
from sklearn.metrics import accuracy_score
import shutil
import json

# Function to standardize images
def standardize_image(image):
    # Redimensiona para 200x200 e converte para escala de cinza
    image = cv2.resize(image, (200, 200), interpolation=cv2.INTER_LANCZOS4)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.equalizeHist(image)
    return image

# Function to apply data augmentation (random rotation)
def augment_data(images):
    augmented_images = []
    for image in images:
        angle = np.random.randint(-10, 10)
        M = cv2.getRotationMatrix2D((image.shape[1]//2, image.shape[0]//2), angle, 1.0)
        augmented_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
        augmented_images.append(augmented_image)
    return augmented_images

# Directories
faces_path_captured = "training/images/captured_faces/"
faces_path_treino = "training/images/training/"
os.makedirs(faces_path_treino, exist_ok=True)

# Organize images captured in training
list_of_faces_captured = [f for f in os.listdir(faces_path_captured) if os.path.isfile(os.path.join(faces_path_captured, f))]

for file in list_of_faces_captured:
    parts = file.split('_')
    if len(parts) >= 2:
        subject = parts[0]
        number = int(parts[2].split('.')[0])
        if number < 50:
            shutil.copyfile(os.path.join(faces_path_captured, file), os.path.join(faces_path_treino, file))

# List image files
list_of_faces_train = [f for f in os.listdir(faces_path_treino) if os.path.isfile(os.path.join(faces_path_treino, f))]

# Prepare training data
training_data, subjects = [], []

for file in list_of_faces_train:
    image_path = os.path.join(faces_path_treino, file)
    image = cv2.imread(image_path)
    image = standardize_image(image)
    image = cv2.equalizeHist(image)
    training_data.append(image)
    subject = file.split('_')[0]
    subjects.append(subject)

# Map subject names to integers
unique_subjects = list(set(subjects))
subject_map = {name: idx + 1 for idx, name in enumerate(unique_subjects)}

# Save the mapping to a JSON file
with open("training/models/subject_map.json", "w") as f:
    json.dump(subject_map, f)

# Convert subject names to integers
subjects = np.array([subject_map[name] for name in subjects], dtype=np.int32)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
X_train, X_test, y_train, y_test = train_test_split(training_data, subjects, test_size=0.2, random_state=42)

# Train LBPH model
model_lbph = cv2.face.LBPHFaceRecognizer_create()
model_lbph.train(X_train, np.array(y_train))

# Testing the LBPH recognizer with test data
predictions = []
for test_img in X_test:
    label, confidence = model_lbph.predict(test_img)
    predictions.append(label)

# Calculate facial recognition accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")


# Save the trained model
model_dir = "training/models"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
model_path = os.path.join(model_dir, "model_lbph.yml")
model_lbph.save(model_path)