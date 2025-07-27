import cv2
import mediapipe as mp
import numpy as np
from constants import * 
from utils import calculate_angle

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(laptopWebcamNumber)

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
      right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
      right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
      right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
      angle = calculate_angle(
        [right_shoulder.x, right_shoulder.y],
        [right_elbow.x, right_elbow.y],
        [right_wrist.x, right_wrist.y]
      )

      text_location = tuple(np.multiply([right_elbow.x, right_elbow.y], [feed_width, feed_height]).astype(int))
      font_size = 5
      thickness = 4
      cv2.putText(image, str(angle),
        text_location,
        cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), thickness, cv2.LINE_AA
        )

    except Exception as e:
      # print(e)
      pass

    # Render detections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
      dot_drawing_spec,
      connection_drawing_spec
    )

    cv2.imshow('Mediapipe Feed', image)

    # Exit on q
    if cv2.waitKey(10) & 0xFF == ord('q'):
      break

cap.release()
cv2.destroyAllWindows()


