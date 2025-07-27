import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

webcamNumber = 1

cap = cv2.VideoCapture(webcamNumber)
while cap.isOpened():
  ret, frame = cap.read()
  cv2.imshow('Mediapipe Feed', frame)

cap.release()
cv2.destroyAllWindows()
