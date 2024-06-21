import cv2
import os
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import shutil
import json

# Function to standardize images
def standardize_image(image):
    image = cv2.resize(image, (200, 200), interpolation=cv2.INTER_LANCZOS4)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.equalizeHist(image)
    return image

# Function to normalize images
def normalize_image(image):
    return image / 255.0

# Function to apply data augmentation (random rotation)
def augment_data(images):
    augmented_images = []
    for image in images:
        angle = np.random.randint(-10, 10)
        M = cv2.getRotationMatrix2D((image.shape[1] // 2, image.shape[0] // 2), angle, 1.0)
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

# data balancing
training_data_bal, subjects_bal = [], []
min_count = min([subjects.tolist().count(s) for s in np.unique(subjects)])
for s in np.unique(subjects):
    data_class = [training_data[i] for i in range(len(subjects)) if subjects[i] == s]
    data_class_bal = resample(data_class, replace=False, n_samples=min_count, random_state=42)
    training_data_bal.extend(data_class_bal)
    subjects_bal.extend([s] * min_count)

# data normalization
training_data_bal = [normalize_image(image) for image in training_data_bal]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(training_data_bal, subjects_bal, test_size=0.2, random_state=42)

# Train LBPH model with hyperparameters
parametros_lbph = {
    'radius': 1,
    'neighbors': 8,
    'grid_x': 8,
    'grid_y': 8
}
model_lbph = cv2.face.LBPHFaceRecognizer_create(**parametros_lbph)
model_lbph.train(X_train, np.array(y_train))

# Train Eigenface model with hyperparameters
parametros_eigenface = {
    'num_components': 50  # Número de componentes principais
}
model_eigenface = cv2.face.EigenFaceRecognizer_create(**parametros_eigenface)
model_eigenface.train(X_train, np.array(y_train))

# Train Fisherface model with hyperparameters
parametros_fisherface = {
    'num_components': 50  # Número de componentes discriminantes
}
model_fisherface = cv2.face.FisherFaceRecognizer_create(**parametros_fisherface)
model_fisherface.train(X_train, np.array(y_train))

# Test models
def test_model(model, X_test):
    predictions = []
    for test_img in X_test:
        label, confidence = model.predict(test_img)
        predictions.append(label)
    return predictions

# Evaluate LBPH model
predictions_lbph = test_model(model_lbph, X_test)
accuracy_lbph = accuracy_score(y_test, predictions_lbph)
print(f"Accuracy LBPH: {accuracy_lbph}")

# Evaluate Eigenface model
predictions_eigenface = test_model(model_eigenface, X_test)
accuracy_eigenface = accuracy_score(y_test, predictions_eigenface)
print(f"Accuracy Eigenface: {accuracy_eigenface}")

# Evaluate Fisherface model
predictions_fisherface = test_model(model_fisherface, X_test)
accuracy_fisherface = accuracy_score(y_test, predictions_fisherface)
print(f"Accuracy Fisherface: {accuracy_fisherface}")

# Save models
model_dir = "training/models"
os.makedirs(model_dir, exist_ok=True)
model_lbph.save(os.path.join(model_dir, "model_lbph.yml"))
model_eigenface.save(os.path.join(model_dir, "model_eigenface.yml"))
model_fisherface.save(os.path.join(model_dir, "model_fisherface.yml"))
