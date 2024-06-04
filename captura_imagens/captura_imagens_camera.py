import cv2
import mediapipe as mp
import os

# Diretório para salvar imagens capturadas
capture_dir = "../treinamento/imagens/captured_faces/"
if not os.path.exists(capture_dir):
    os.makedirs(capture_dir)

cap = cv2.VideoCapture(0)
counter = 0

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)

if not cap.isOpened():
    print("Erro ao abrir a webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro ao capturar frame")
        break
    
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)

    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(frame, faceLms, mpFaceMesh.FACEMESH_CONTOURS, drawSpec, drawSpec)
            
            # Salvar a imagem do rosto detectado
            rosto_path = os.path.join(capture_dir, f"face_{counter}.jpg")
            cv2.imwrite(rosto_path, frame)
            counter += 1

    cv2.imshow('Webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
