import cv2
import time
import torch
from gravity import get_gravity_vector
from pose_utils import *

from ultralytics import YOLO

WRIST_IDX = 10
ELBOW_IDX = 8

def main():

    if torch.cuda.is_available():
        device = "cuda"
        print(f"[GPU] Using {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("[CPU] CUDA not available — install CUDA-enabled PyTorch for GPU.")
        print("      pip uninstall torch -y")
        print("      pip install torch --index-url https://download.pytorch.org/whl/cu121")

    pose_model = YOLO("yolo11x-pose.pt")
    pose_model.to(device)
    hand_model = YOLO("yolo11n-pose.pt")
    hand_model.to(device)

    #initialize variables needed
    camera_index = 0
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open any camera.")

    accelerometer_data = None
    gravity_vec = get_gravity_vector(accelerometer_data)

    VISIBILITY_THRESHOLD = 0.4      # confidence. lower for noisy cameras, higher for better cameras
    PERPENDICULAR_TOLERANCE = 10

    camera_tilt_deg = 0.0
    TILT_STEP = 1.0

    fade_message = ""
    fade_start_time = 0
    fade_duration = 2.0

    prev_time = time.time()
    fps = 0.0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape

        now = time.time()
        dt = now - prev_time
        prev_time = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)

        if camera_tilt_deg != 0.0:
            M = cv2.getRotationMatrix2D((w / 2, h / 2), camera_tilt_deg, 1.0)
            frame = cv2.warpAffine(frame, M, (w, h))

        # run pose model
        results = pose_model(frame, verbose=False, device=device)[0]

        wx = wy = wz = ex = ey = None
        detector_label = "NONE"

        if results.keypoints is not None and len(results.keypoints) > 0:
            kps = results.keypoints[0]

            kp_data = kps.data[0].cpu().numpy()

            wrist_x, wrist_y, wrist_conf = kp_data[WRIST_IDX]
            elbow_x, elbow_y, elbow_conf = kp_data[ELBOW_IDX]

            # check if confidence is above the set threshold
            if wrist_conf >= VISIBILITY_THRESHOLD and elbow_conf >= VISIBILITY_THRESHOLD:
                detector_label = "POSE"
                wx, wy = int(wrist_x), int(wrist_y)
                ex, ey = int(elbow_x), int(elbow_y)
                wz = 0

        if wx is not None:
            # get forearm points which is middle of wrist and elbow
            fx, fy = (wx + ex) // 2, (wy + ey) // 2

            # math the angle using forearm vector and gravity vector
            angle = forearm_angle((wx, wy), (ex, ey), gravity_vec)

            # check if within limits set
            is_perpendicular = angle <= PERPENDICULAR_TOLERANCE
            perp_status = "PERPENDICULAR" if is_perpendicular else f"OFF BY {angle:.1f} deg"
            perp_color = (0, 255, 0) if is_perpendicular else (0, 0, 255)

            is_in_line, is_in_depth = wrist_in_line(wx, wz, w)

            x_status = "IN LINE" if is_in_line else "NOT IN LINE"
            x_color = (0, 255, 0) if is_in_line else (0, 0, 255)

            z_status = "GOOD DISTANCE" if is_in_depth else "BAD DISTANCE"
            z_color = (0, 255, 0) if is_in_depth else (0, 0, 255)

            # draw the points and lines
            cv2.circle(frame, (wx, wy), 10, (0, 255, 0),   -1)
            cv2.circle(frame, (ex, ey), 10, (255, 0, 255), -1)
            cv2.circle(frame, (fx, fy),  8, (255, 255, 0), -1)
            cv2.line(frame, (wx, wy), (ex, ey), (255, 255, 255), 3)

            dx, dy = ex - wx, ey - wy
            ext_f = (wx + int(dx * 2), wy + int(dy * 2))
            ext_w = (wx - int(dx * 1), wy - int(dy * 1))
            cv2.line(frame, (ex, ey), ext_f, (200, 200, 200), 1)
            cv2.line(frame, (wx, wy), ext_w, (200, 200, 200), 1)

            # write stats on on screen
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
            cv2.putText(frame, "Not detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        badge_color = (255, 180, 0) if detector_label == "POSE" else (0, 0, 200)
        cv2.rectangle(frame, (w - 140, 10), (w - 10, 45), badge_color, -1)
        cv2.putText(frame, detector_label, (w - 130, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

        cv2.putText(frame, f"{fps:.1f} FPS  [{device.upper()}]",
                    (w - 220, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 255), 2)

        cv2.putText(frame, f"Cam Tilt: {camera_tilt_deg:.1f} deg",
                    (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 255), 2)

        # write control instructions
        help_lines = [
            "]: Tilt CW",
            "[: Tilt CCW",
            "C: Switch Cam",
            "ESC: Exit"
        ]
        y0 = h - 20
        for i, text in enumerate(help_lines[::-1]):
            cv2.putText(frame, text,
                        (w - 200, y0 - i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        if fade_message:
            current_time = cv2.getTickCount() / cv2.getTickFrequency()
            elapsed = current_time - fade_start_time
            if elapsed < fade_duration:
                alpha = 1.0 - (elapsed / fade_duration)
                overlay = frame.copy()
                cv2.putText(overlay, fade_message, (50, h - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            else:
                fade_message = ""

        cv2.imshow("Forearm Tracker", frame)

        # inputs
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == ord('['):
            camera_tilt_deg -= TILT_STEP
            fade_message = f"Tilt: {camera_tilt_deg:.1f} deg"
            fade_start_time = cv2.getTickCount() / cv2.getTickFrequency()
        elif key == ord(']'):
            camera_tilt_deg += TILT_STEP
            fade_message = f"Tilt: {camera_tilt_deg:.1f} deg"
            fade_start_time = cv2.getTickCount() / cv2.getTickFrequency()
        elif key in (ord('c'), ord('C')):
            new_index = 1 - camera_index
            new_cap = cv2.VideoCapture(new_index)
            if not new_cap.isOpened():
                fade_message = "No other camera found"
                fade_start_time = cv2.getTickCount() / cv2.getTickFrequency()
                new_cap.release()
            else:
                cap.release()
                cap = new_cap
                camera_index = new_index

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
