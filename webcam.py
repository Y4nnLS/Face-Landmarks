import cv2
import mediapipe as mp
import time

cap=cv2.VideoCapture(0)
pTime = 0

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
drawnSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)


if not cap.isOpened():
    print("Erro ao abrir a webcam")
    exit()

while True:
    ret,frame = cap.read()
    if not ret:
        print("Erro ao capturar frame")
        break
    
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)

    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(frame, faceLms, mpFaceMesh.FACEMESH_CONTOURS, drawnSpec, drawnSpec)

            for id, lm in enumerate(faceLms.landmark):
                # print(lm)
                fh, fw, fc = frame.shape
                x, y = int(lm.x*fw), int(lm.y*fh)


    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(frame, f'FPS: {int(fps)}', (10,30),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2, cv2.LINE_AA)
    cv2.imshow('Webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()