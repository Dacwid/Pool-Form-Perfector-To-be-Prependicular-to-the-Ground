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
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
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

# Pose Model
pose_options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="models/pose_landmarker_full.task"),
    running_mode=RunningMode.VIDEO,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
pose_detector = PoseLandmarker.create_from_options(pose_options)

# Camera Setup
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open any camera. Check your camera connection.")

# Gravity Setup
accelerometer_data = None
gravity_vec = get_gravity_vector(accelerometer_data)

# Landmark Indices
WRIST_IDX = 0          # Hand landmarker wrist
RIGHT_ELBOW_IDX = 14   # Pose landmarker right elbow

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

    # Detectors activated
    hand_results = hand_detector.detect_for_video(mp_image, frame_timestamp_ms)
    pose_results = pose_detector.detect_for_video(mp_image, frame_timestamp_ms)

    # Check for requirements
    hand_found = hand_results.hand_landmarks and len(hand_results.hand_landmarks) > 0
    pose_found = (pose_results.pose_landmarks and len(pose_results.pose_landmarks) > 0
                  and len(pose_results.pose_landmarks[0]) > RIGHT_ELBOW_IDX)

    if hand_found and pose_found:
        wrist_lm = hand_results.hand_landmarks[0][WRIST_IDX]
        elbow_lm = pose_results.pose_landmarks[0][RIGHT_ELBOW_IDX]

        wx, wy, wz = get_pixel_coordinates(wrist_lm, w, h)
        ex, ey, ez = get_pixel_coordinates(elbow_lm, w, h)

        # Check wrist alignment with camera
        in_line = wrist_in_line(wx, wrist_lm.z, w)
        status = "YES" if in_line else "NO"

        # Check forearm angle
        angle = forearm_angle((ex, ey), (wx, wy), gravity_vec)

        # Draw joints and line
        cv2.circle(frame, (wx, wy), 10, (0, 255, 0), -1)   # wrist green
        cv2.circle(frame, (ex, ey), 10, (255, 0, 0), -1)   # elbow blue
        cv2.line(frame, (wx, wy), (ex, ey), (255, 255, 255), 3)

        # Text
        cv2.putText(frame, f"Wrist: ({wx},{wy})", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Depth Z: {wz:.2f}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"In Line: {status}", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"Forearm Angle: {angle:.1f} deg", (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    elif not hand_found and not pose_found:
        cv2.putText(frame, "No hand or pose detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    elif not hand_found:
        cv2.putText(frame, "No hand detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    elif not pose_found:
        cv2.putText(frame, "No pose/elbow detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow("Forearm Tracker", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()