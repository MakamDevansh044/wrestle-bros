#this code works with a external webcam (like phones , where the video invertion takes place )
# if you wants to run with laptop cam change 1 to 0 in 13th line , and change the inversion part in the code 

import cv2
import mediapipe as mp
import numpy as np
import time
import pyautogui

# === Pose Setup ===
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
cap = cv2.VideoCapture(1)    

# === Globals ===
initial_hip_center = None
initial_forward_vec = None
initial_right_vec = None
calibrated = False

# === Thresholds ===
THRESH_HIP_Y = 0.05
THRESH_PUNCH = 0.30
THRESH_MOVE = 0.15

# === Gesture Timing ===
active_keys = {}
display_texts = {}

KEY_MAP = {
    "Jump": "up",
    "Down": "down",
    "Left": "left",
    "Right": "right",
    "Front": "left",
    "Back": "right",
    "Punch": "space",
}

HOLD_TIME = {
    "Punch": 0.5,
    "Jump": 0,
    "Down": 0,
    "Left": 0,
    "Right": 0,
    "Front": 0,
    "Back": 0,
}

DISPLAY_DURATION = 0.8  # seconds


def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm != 0 else v


def trigger_key(gesture):
    key = KEY_MAP[gesture]
    if gesture not in active_keys:
        pyautogui.keyDown(key)
        active_keys[gesture] = time.time()
        display_texts[gesture] = time.time()


def release_keys():
    now = time.time()
    for gesture in list(active_keys):
        hold_limit = HOLD_TIME.get(gesture, 0)
        if hold_limit > 0:
            if now - active_keys[gesture] >= hold_limit:
                pyautogui.keyUp(KEY_MAP[gesture])
                del active_keys[gesture]
        else:
            if gesture not in current_gestures:
                pyautogui.keyUp(KEY_MAP[gesture])
                del active_keys[gesture]


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)
    h, w, _ = frame.shape
    current_gestures = []

    if result.pose_landmarks:
        landmarks = result.pose_landmarks.landmark

        # === Swapped left/right ===
        nose = np.array([landmarks[0].x, landmarks[0].y, landmarks[0].z])
        left_wrist = np.array([landmarks[16].x, landmarks[16].y, landmarks[16].z])  # right
        right_wrist = np.array([landmarks[15].x, landmarks[15].y, landmarks[15].z])  # left
        left_hip = np.array([landmarks[24].x, landmarks[24].y, landmarks[24].z])
        right_hip = np.array([landmarks[23].x, landmarks[23].y, landmarks[23].z])
        left_foot = np.array([landmarks[32].x, landmarks[32].y, landmarks[32].z])
        right_foot = np.array([landmarks[31].x, landmarks[31].y, landmarks[31].z])

        hip_center = (left_hip + right_hip) / 2

        if calibrated:
            # === Punch ===
            if right_wrist[2] < nose[2] - THRESH_PUNCH or left_wrist[2] < nose[2] - THRESH_PUNCH:
                current_gestures.append("Punch")
                trigger_key("Punch")

            # === Jump / Down ===
            if hip_center[1] < initial_hip_center[1] - THRESH_HIP_Y:
                current_gestures.append("Jump")
                trigger_key("Jump")
            elif hip_center[1] > initial_hip_center[1] + THRESH_HIP_Y:
                current_gestures.append("Down")
                trigger_key("Down")

            # === Forward / Back ===
            delta_z = hip_center[2] - initial_hip_center[2]
            if delta_z < -THRESH_MOVE:
                current_gestures.append("Front")
                trigger_key("Front")
            elif delta_z > THRESH_MOVE:
                current_gestures.append("Back")
                trigger_key("Back")

            # === Left / Right ===
            hip_2D = hip_center[[0, 2]]
            move_vec = hip_2D - initial_hip_center[[0, 2]]
            right_proj = np.dot(move_vec, initial_right_vec)

            if right_proj > THRESH_MOVE:
                current_gestures.append("Right")
                trigger_key("Right")
            elif right_proj < -THRESH_MOVE:
                current_gestures.append("Left")
                trigger_key("Left")

        # Draw pose
        mp.solutions.drawing_utils.draw_landmarks(
            frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # === Text Display ===
    now = time.time()
    display_texts = {g: t for g, t in display_texts.items() if now - t <= DISPLAY_DURATION}
    y = 40
    for g in display_texts:
        cv2.putText(frame, f"- {g}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        y += 30

    if not calibrated:
        cv2.putText(frame, "Press 'R' to calibrate front direction", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow("Gesture Controller", frame)
    key = cv2.waitKey(1) & 0xFF

    # === Calibration ===
    if key == ord('r') and result.pose_landmarks:
        left_hip_2D = np.array([landmarks[24].x, landmarks[24].z])
        right_hip_2D = np.array([landmarks[23].x, landmarks[23].z])
        hip_vec = right_hip_2D - left_hip_2D

        forward_vec = normalize(np.array([-hip_vec[1], hip_vec[0]]))
        right_vec = normalize(hip_vec)

        initial_forward_vec = forward_vec
        initial_right_vec = right_vec
        initial_hip_center = (left_hip + right_hip) / 2
        calibrated = True
        print("✅ Calibration complete.")

    elif key == ord('q'):
        break

    release_keys()

cap.release()
cv2.destroyAllWindows()
