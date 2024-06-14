import cv2
import mediapipe as mp
import os
import time

# Diretório para salvar imagens capturadas
capture_dir = "treinamento/imagens/captured_faces/"
if not os.path.exists(capture_dir):
    os.makedirs(capture_dir)

# cap = cv2.VideoCapture(0)

# mpDraw = mp.solutions.drawing_utils
# mpFaceMesh = mp.solutions.face_mesh
# faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
# drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)

# if not cap.isOpened():
#     print("Erro ao abrir a webcam")
#     exit()

# Inicializar temporizador
capture_interval = 0.3  # intervalo de captura em segundos

while True:
    sujeito = input("Por favor, insira o nome do sujeito (ou 'exit' para sair): ")
    if sujeito.lower() == 'exit':
        break
    cap = cv2.VideoCapture(0)

    mpDraw = mp.solutions.drawing_utils
    mpFaceMesh = mp.solutions.face_mesh
    faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
    drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)
    
    if not cap.isOpened():
        print("Erro ao abrir a webcam")
        exit()
    
    # sujeito = int(sujeito)
    counter = 0
    start_time = time.time()

    while counter < 50:  # Capturar 20 imagens por sujeito
        ret, frame = cap.read()
        if not ret:
            print("Erro ao capturar frame")
            break
        
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = faceMesh.process(imgRGB)

        if results.multi_face_landmarks:
            for faceLms in results.multi_face_landmarks:
                mpDraw.draw_landmarks(frame, faceLms, mpFaceMesh.FACEMESH_CONTOURS, drawSpec, drawSpec)
                
                # Determinar a região do rosto e salvar a imagem
                ih, iw, ic = frame.shape
                face_coords = [(int(lm.x * iw), int(lm.y * ih)) for lm in faceLms.landmark]
                x_coords = [coord[0] for coord in face_coords]
                y_coords = [coord[1] for coord in face_coords]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                
                # Verificar se é hora de capturar uma nova imagem
                if time.time() - start_time >= capture_interval:
                    rosto = frame[y_min:y_max, x_min:x_max]
                    rosto_path = os.path.join(capture_dir, f"{sujeito}_face_{counter}.jpg")
                    cv2.imwrite(rosto_path, rosto)
                    print(f"Imagem {counter+1} salva em {rosto_path}")
                    counter += 1
                    start_time = time.time()  # Reiniciar temporizador

        cv2.imshow('Webcam', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
# cap.release()
# cv2.destroyAllWindows()
