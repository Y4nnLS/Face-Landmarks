import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture("Videos/2.mp4")
pTime = 0

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh()

while True:
    sucess, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)




    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20,70),cv2.FONT_HERSHEY_PLAIN, 3,(0,255,0),3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)