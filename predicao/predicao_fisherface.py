import cv2
import mediapipe as mp
import numpy as np
import time
import os
import json

# Função para padronizar as imagens
def padronizar_imagem(imagem):
    imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    imagem = cv2.resize(imagem, (200, 200), interpolation=cv2.INTER_LANCZOS4)
    return imagem

# Carregar o modelo LBPH treinado
modelo_path = "treinamento/modelos/modelo_fisherface.yml"
modelo_lbph = cv2.face.FisherFaceRecognizer_create()
modelo_lbph.read(modelo_path)

# Carregar o mapeamento de sujeitos
with open("treinamento/modelos/sujeito_map.json", "r") as f:
    sujeito_map = json.load(f)
reverse_sujeito_map = {v: k for k, v in sujeito_map.items()}

# Inicializar a captura de vídeo
cap = cv2.VideoCapture(0)

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
            # mpDraw.draw_landmarks(frame, faceLms, mpFaceMesh.FACEMESH_CONTOURS, drawSpec, drawSpec)
            
            # Determinar a região do rosto e fazer a predição
            ih, iw, ic = frame.shape
            face_coords = [(int(lm.x * iw), int(lm.y * ih)) for lm in faceLms.landmark]
            x_coords = [coord[0] for coord in face_coords]
            y_coords = [coord[1] for coord in face_coords]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            rosto = frame[y_min:y_max, x_min:x_max]
            rosto_padronizado = padronizar_imagem(rosto)
            
            predicao, confianca = modelo_lbph.predict(rosto_padronizado)
            nome_sujeito = reverse_sujeito_map.get(predicao, "Desconhecido")
            
            # Mostrar o nome do sujeito no frame
            # cv2.putText(frame, predicao, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, nome_sujeito, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    cv2.imshow('Webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
