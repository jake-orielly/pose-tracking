import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils

dot_drawing_spec = mp_drawing.DrawingSpec(color=(66, 245, 66), thickness=8, circle_radius=2)
connection_drawing_spec = mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=3, circle_radius=2)

laptopWebcamNumber = 0
phoneWebcamNumber = 1
