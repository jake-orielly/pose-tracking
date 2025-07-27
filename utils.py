import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose


def calculate_landmarks_angle(start_joint, mid_joint, end_joint, landmarks):
    start_landmark = landmarks[mp_pose.PoseLandmark[start_joint].value]
    mid_landmark = landmarks[mp_pose.PoseLandmark[mid_joint].value]
    end_landmark = landmarks[mp_pose.PoseLandmark[end_joint].value]
    return calculate_angle(
        [start_landmark.x, start_landmark.y],
        [mid_landmark.x, mid_landmark.y],
        [end_landmark.x, end_landmark.y]
    )

def calculate_angle(a, b, c):
  a = np.array(a)
  b = np.array(b)
  c = np.array(c)

  radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
  angle = np.abs(radians*180.0/np.pi)

  if angle > 180.0:
    angle = 360 - angle

  return angle 