import cv2
import numpy as np
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = model.process(image_rgb)
    image_rgb.flags.writeable = True
    return results

def extract_landmarks(results) -> np.ndarray:
    # 33 pose landmarks (x, y, z, visibility) -> 132 values
    if results.pose_landmarks:
        pose_raw = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark])
        # Center relative to nose landmark (index 0)
        center = pose_raw[0, :3]
        pose_raw[:, :3] = pose_raw[:, :3] - center
        pose = pose_raw.flatten()
    else:
        pose = np.zeros(33 * 4)

    # 21 left hand landmarks (x, y, z) -> 63 values
    if results.left_hand_landmarks:
        lh_raw = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark])
        # Center relative to left wrist (index 0)
        lh_wrist = lh_raw[0]
        lh_raw = lh_raw - lh_wrist
        lh = lh_raw.flatten()
    else:
        lh = np.zeros(21 * 3)

    # 21 right hand landmarks (x, y, z) -> 63 values
    if results.right_hand_landmarks:
        rh_raw = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark])
        # Center relative to right wrist (index 0)
        rh_wrist = rh_raw[0]
        rh_raw = rh_raw - rh_wrist
        rh = rh_raw.flatten()
    else:
        rh = np.zeros(21 * 3)

    return np.concatenate([pose, lh, rh])