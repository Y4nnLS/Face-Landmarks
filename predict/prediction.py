import cv2
import mediapipe as mp
import json
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

# Function to standardize images
def standardize_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (200, 200), interpolation=cv2.INTER_LANCZOS4)
    image = cv2.equalizeHist(image)
    return image

def predict(selected_option):
    # Load the trained LBPH model
    match selected_option:
        case "1":
            model_path = "training/models/model_lbph.yml"
            model = cv2.face.LBPHFaceRecognizer_create()
            model.read(model_path)
        case "2":
            model_path = "training/models/model_fisherface.yml"
            model = cv2.face.LBPHFaceRecognizer_create()
            model.read(model_path)
        case "3":
            model_path = "training/models/modelo_eigenface.yml"
            model = cv2.face.EigenFaceRecognizer_create()
            model.read(model_path)

    # Load the individual mapping
    with open("training/models/subject_map.json", "r") as f:
        individual_map = json.load(f)
        reverse_individual_map = {v: k for k, v in individual_map.items()}

    # Initialize video capture
    cap = cv2.VideoCapture(0)

    mpDraw = mp.solutions.drawing_utils
    mpFaceMesh = mp.solutions.face_mesh
    faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
    drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)

    if not cap.isOpened():
        print("Error opening webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error capturing frame")
            break

        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = faceMesh.process(imgRGB)

        if results.multi_face_landmarks:
            for faceLms in results.multi_face_landmarks:

                # Determine the region of the face and make the prediction
                ih, iw, ic = frame.shape
                face_coords = [(int(lm.x * iw), int(lm.y * ih)) for lm in faceLms.landmark]
                x_coords = [coord[0] for coord in face_coords]
                y_coords = [coord[1] for coord in face_coords]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                
                # Adjust the face area to ensure that only the face is considered
                margin = 40  # Additional margin around the face
                x_min = max(0, x_min - margin)
                y_min = max(0, y_min - margin)
                x_max = min(iw, x_max + margin)
                y_max = min(ih, y_max + margin)
                
                face = frame[y_min:y_max, x_min:x_max]
                standard_face = standardize_image(face)
                face_equalized = cv2.equalizeHist(standard_face)
                
                # Make prediction with the trained model
                predict, trust = model.predict(face_equalized)
                
                # Set a trust threshold to identify unknowns
                threshold_trust = 110
                trust = round(trust)
                print(f"Prediction: {predict}, Trust: {trust}")
                
                # Check if the prediction is valid or if it is an unknown
                if trust < threshold_trust:
                    individual_name = reverse_individual_map.get(predict, "Unknown")
                else:
                    individual_name = "Unknown"
                
                # Show the individual's name in the frame and the message of info
                cv2.putText(frame, individual_name, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, 'Press Q to close the Webcam', (30, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        cv2.imshow('Webcam', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    predict()