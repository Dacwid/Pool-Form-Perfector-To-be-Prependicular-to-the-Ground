import cv2
from gravity import get_gravity_vector
from pose_utils import *

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# MediaPipe Setup
BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
RunningMode = vision.RunningMode

# Hand Model
hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="models/hand_landmarker.task"),
    running_mode=RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
hand_detector = HandLandmarker.create_from_options(hand_options)

# Camera Setup
camera_index = 0
cap = cv2.VideoCapture(camera_index)
if not cap.isOpened():
    raise RuntimeError("Could not open any camera.")

# Gravity Setup
accelerometer_data = None
gravity_vec = get_gravity_vector(accelerometer_data)

# Constants
WRIST_IDX = 0
FOREARM_REF_IDX = 18
PERPENDICULAR_TOLERANCE = 10

fade_message = ""
fade_start_time = 0
fade_duration = 2.0

# Timestamp Setup
frame_timestamp_ms = 0
fps = cap.get(cv2.CAP_PROP_FPS) or 30

# Main Loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_timestamp_ms += int(1000 / fps)

    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    hand_results = hand_detector.detect_for_video(mp_image, frame_timestamp_ms)

    # Filter for right hand only
    right_hand_landmarks = None
    if hand_results.hand_landmarks and hand_results.handedness:
        for i, handedness in enumerate(hand_results.handedness):
            if handedness[0].category_name == "Right":
                right_hand_landmarks = hand_results.hand_landmarks[i]
                break

    if right_hand_landmarks:
        wrist_lm = right_hand_landmarks[WRIST_IDX]
        ref_lm = right_hand_landmarks[FOREARM_REF_IDX]

        wx, wy, wz = get_pixel_coordinates(wrist_lm, w, h)
        rx, ry, rz = get_pixel_coordinates(ref_lm, w, h)

        angle = forearm_angle((rx, ry), (wx, wy), gravity_vec)

        is_perpendicular = angle <= PERPENDICULAR_TOLERANCE
        perp_status = "PERPENDICULAR" if is_perpendicular else f"OFF BY {angle:.1f} deg"
        perp_color = (0, 255, 0) if is_perpendicular else (0, 0, 255)

        is_in_line, is_in_depth = wrist_in_line(wx, wz, w)

        x_status = "IN LINE" if is_in_line else "NOT IN LINE"
        x_color = (0, 255, 0) if is_in_line else (0, 0, 255)

        z_status = "GOOD DISTANCE" if is_in_depth else "BAD DISTANCE"
        z_color = (0, 255, 0) if is_in_depth else (0, 0, 255)

        # Draw joints
        cv2.circle(frame, (wx, wy), 10, (0, 255, 0), -1)
        cv2.circle(frame, (rx, ry), 8, (255, 255, 0), -1)
        cv2.line(frame, (wx, wy), (rx, ry), (255, 255, 255), 3)

        # Line extension
        dx, dy = rx - wx, ry - wy
        ex_f, ey_f = wx + int(dx * 2), wy + int(dy * 2)
        ex_w, ey_w = wx - int(dx * 3), wy - int(dy * 3)
        cv2.line(frame, (rx, ry), (ex_f, ey_f), (200, 200, 200), 1)
        cv2.line(frame, (wx, wy), (ex_w, ey_w), (200, 200, 200), 1)

        # Text
        cv2.putText(frame, f"Forearm Angle: {angle:.1f} deg", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(frame, perp_status, (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, perp_color, 3)
        cv2.putText(frame, f"Wrist: ({wx},{wy})", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, x_status, (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, x_color, 3)
        cv2.putText(frame, z_status, (20, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, z_color, 3)


    else:
        cv2.putText(frame, "Right hand not detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    help_lines = [
        "1: Pinky",
        "2: Index",
        "3: Thumb",
        "C: Switch Cam",
        "ESC: Exit"
    ]

    y0 = h - 20
    for i, text in enumerate(help_lines[::-1]):
        cv2.putText(frame, text,
                    (w - 180, y0 - i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (200, 200, 200),
                    2)

    # show frame
    if fade_message:
        current_time = cv2.getTickCount() / cv2.getTickFrequency()
        elapsed = current_time - fade_start_time

        if elapsed < fade_duration:
            alpha = 1.0 - (elapsed / fade_duration)

            overlay = frame.copy()
            cv2.putText(overlay, fade_message,
                        (50, h - 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 0, 255),
                        3)

            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        else:
            fade_message = ""

    cv2.imshow("Forearm Tracker", frame)

    # Key presses

    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        break
    elif key == ord('1'):
        FOREARM_REF_IDX = 18
    elif key == ord('2'):
        FOREARM_REF_IDX = 5
    elif key == ord('3'):
        FOREARM_REF_IDX = 3
    elif key in (ord('c'), ord('C')):
        new_index = 1 - camera_index
        new_cap = cv2.VideoCapture(new_index)

        if not new_cap.isOpened():
            # Trigger fade message
            fade_message = "No other camera found"
            fade_start_time = cv2.getTickCount() / cv2.getTickFrequency()
            new_cap.release()
        else:
            cap.release()
            cap = new_cap
            camera_index = new_index

cap.release()
cv2.destroyAllWindows()