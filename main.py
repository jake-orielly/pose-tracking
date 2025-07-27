import cv2
import mediapipe as mp
import numpy as np
from constants import * 
from utils import calculate_angle, calculate_landmarks_angle

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(laptopWebcamNumber)

left_curl_count = 0
left_curl_stage = None 

right_curl_count = 0
right_curl_stage = None

right_shoulder_count = 0
right_shoulder_stage = None

left_shoulder_count = 0
left_shoulder_stage = None


feed_width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
feed_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    ret, frame = cap.read()

    # Recolor image to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Make detection
    results = pose.process(image)

    # Recolor back to BGR
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Extract landmarks
    try:
      landmarks = results.pose_landmarks.landmark

      right_elbow_angle = calculate_landmarks_angle(
        'RIGHT_SHOULDER', 'RIGHT_ELBOW', 'RIGHT_WRIST',
        landmarks
      )
      left_elbow_angle = calculate_landmarks_angle(
        'LEFT_SHOULDER', 'LEFT_ELBOW', 'LEFT_WRIST',
        landmarks
      )

      right_arm_angle = calculate_landmarks_angle(
        'RIGHT_HIP', 'RIGHT_SHOULDER', 'RIGHT_ELBOW',
        landmarks
      )
      left_arm_angle = calculate_landmarks_angle(
        'LEFT_HIP', 'LEFT_SHOULDER', 'LEFT_ELBOW',
        landmarks
      )

      if left_elbow_angle > 160:
        left_curl_stage = "down"
      elif left_elbow_angle < 30 and left_curl_stage == "down":
        left_curl_stage = "up"
        left_curl_count += 1

      if right_elbow_angle > 160:
        right_curl_stage = "down"
      elif right_elbow_angle < 30 and right_curl_stage == "down":
        right_curl_stage = "up"
        right_curl_count += 1

      if right_arm_angle < 90:
        right_shoulder_stage = "down"
      elif right_arm_angle > 170 and right_shoulder_stage == "down":
        right_shoulder_stage = "up"
        right_shoulder_count += 1

      if left_arm_angle < 90:
        left_shoulder_stage = "down"
      elif left_arm_angle > 170 and left_shoulder_stage == "down":
        left_shoulder_stage = "up"
        left_shoulder_count += 1

      font_size = 3
      thickness = 4

      cv2.rectangle(image, (0,0), (800,250), (245,117,16), -1)
      cv2.putText(image, 'Curls: ' + str((right_curl_count, left_curl_count)),
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), thickness, cv2.LINE_AA
        )
      cv2.putText(image, 'Presses: ' + str((right_shoulder_count, left_shoulder_count)),
        (20, 200),
        cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), thickness, cv2.LINE_AA
        )

    except Exception as e:
      print(e)
      pass

    cv2.imshow('Mediapipe Feed', image)

    # Exit on q
    if cv2.waitKey(10) & 0xFF == ord('q'):
      break

cap.release()
cv2.destroyAllWindows()


