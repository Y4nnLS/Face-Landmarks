import cv2
import mediapipe as mp
import os
import time

# Directory to save captured images
capture_dir = "./training/images/captured_faces"
if not os.path.exists(capture_dir):
    os.makedirs(capture_dir)

# Initialize timer
capture_interval = 0.2  # Capture interval in seconds

while True:
    subject = input("Please, enter the subject's name (or 'exit' to exit): ")
    if subject.lower() == 'exit':
        break
    cap = cv2.VideoCapture(0)

    mpDraw = mp.solutions.drawing_utils
    mpFaceMesh = mp.solutions.face_mesh
    faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
    drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)
    
    if not cap.isOpened():
        print("Error opening webcam")
        exit()
    
    counter = 0
    start_time = time.time()

    while counter < 50:  # Capture 50 images per subject
        ret, frame = cap.read()
        if not ret:
            print("Error capturing frame")
            break
        
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = faceMesh.process(imgRGB)

        if results.multi_face_landmarks:
            for faceLms in results.multi_face_landmarks:
                # mpDraw.draw_landmarks(frame, faceLms, mpFaceMesh.FACEMESH_CONTOURS, drawSpec, drawSpec)
                
                # Determine the region of the face and save the image
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
                
                # Check if it's time to capture a new image
                if time.time() - start_time >= capture_interval:
                    face = frame[y_min:y_max, x_min:x_max]
                    face_path = os.path.join(capture_dir, f"{subject}_face_{counter}.jpg")
                    cv2.imwrite(face_path, face)
                    print(f"Image {counter+1} save in {face_path}.")
                    counter += 1
                    start_time = time.time()  # Reset timer

        # Show the massage of info
        cv2.putText(frame, 'Press Q to close the Webcam', (30, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
        
        cv2.imshow('Webcam', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): # Key to close Webcam
            break

    cap.release()
    cv2.destroyAllWindows()