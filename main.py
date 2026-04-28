import cv2
import time
import numpy as np
import torch
from gravity import get_gravity_vector
from pose_utils import *
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from ultralytics import YOLO

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

    #initialize variables needed
    camera_index = 0
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open any camera.")

    accelerometer_data = None
    gravity_vec = get_gravity_vector(accelerometer_data)

    # YOLOV11 indexes
    WRIST_IDX = 10
    ELBOW_IDX = 8

    # mediapipe hand landmark indices
    WRIST_LM = 0
    MIDDLE_MCP = 9

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    base_options = mp_python.BaseOptions(model_asset_path="models/hand_landmarker.task")
    hand_options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    hand = vision.HandLandmarker.create_from_options(hand_options)

    VISIBILITY_THRESHOLD = 0.4           # confidence. lower for noisy cameras, higher for better cameras
    PERPENDICULAR_TOLERANCE = 10
    TARGET_FOREARM_PX = 100              # just test for your own environemnt

    GESTURE_HOLD = 0.5                   # seconds to hold gesture
    GESTURE_COOLDOWN = 2.0               # cool down to reuse gesture

    camera_tilt_deg = 0.0
    TILT_STEP = 1.0

    fade_message = ""
    fade_start_time = 0
    fade_duration = 2.0

    prev_time = time.time()
    fps = 0.0

    locked_id = None
    zoom_locked = False
    last_crop = None

    peace_start = None
    thumbs_start = None
    last_gesture_fire = 0.0

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

        # so overlays dont get changed colorwise
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        l_chan = clahe.apply(l_chan)
        proc_frame = cv2.cvtColor(cv2.merge((l_chan, a_chan, b_chan)), cv2.COLOR_LAB2BGR)

        mp_frame = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)

        # run pose model with tracking
        results = pose_model.track(proc_frame, verbose=False, device=device, persist=True)[0]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=mp_frame)
        results_mp = hand.detect_for_video(mp_image, int(now * 1000))

        wx = wy = wz = ex = ey = None
        detector_label = "NONE"

        if (results.boxes is not None
                and results.boxes.id is not None
                and results.keypoints is not None):
            ids = results.boxes.id.cpu().numpy().astype(int)
            boxes = results.boxes.xywh.cpu().numpy()

            # if no one is locked, or our locked person disappeared, pick the one closest to centre
            if locked_id is None or locked_id not in ids:
                dists = [abs(box[0] - w / 2) for box in boxes]
                locked_id = int(ids[int(np.argmin(dists))])

            if locked_id in ids:
                idx = int(np.where(ids == locked_id)[0][0])
                kp_data = results.keypoints[idx].data[0].cpu().numpy()

                wrist_x, wrist_y, wrist_conf = kp_data[WRIST_IDX]
                elbow_x, elbow_y, elbow_conf = kp_data[ELBOW_IDX]

                # check if confidence is above the set threshold
                if wrist_conf >= VISIBILITY_THRESHOLD and elbow_conf >= VISIBILITY_THRESHOLD:
                    detector_label = f"POSE id:{locked_id}"
                    wx, wy = int(wrist_x), int(wrist_y)
                    ex, ey = int(elbow_x), int(elbow_y)
                    wz = 0

        # status values for overlays
        angle = None
        perp_status = perp_color = None
        x_status = x_color = None
        z_status = z_color = None

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

        # zoom happens before text so that text doesnt get clipped off
        if zoom_locked and last_crop is not None:
            cx0, cy0, cw, ch = last_crop
            cropped = frame[cy0:cy0 + ch, cx0:cx0 + cw]
            display_frame = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
        elif wx is not None:
            forearm_len = np.hypot(wx - ex, wy - ey)
            if forearm_len > 1:
                zoom = max(1.0, min(TARGET_FOREARM_PX / forearm_len, 4.0))
            else:
                zoom = 1.0

            cw, ch = int(w / zoom), int(h / zoom)
            cx0 = int(fx - cw / 2)
            cy0 = int(fy - ch / 2)
            cx0 = max(0, min(cx0, w - cw))
            cy0 = max(0, min(cy0, h - ch))

            last_crop = (cx0, cy0, cw, ch)

            cropped = frame[cy0:cy0 + ch, cx0:cx0 + cw]
            display_frame = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            display_frame = frame.copy()

        if wx is not None:
            cv2.putText(display_frame, f"Forearm Angle: {angle:.1f} deg", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(display_frame, perp_status, (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, perp_color, 3)
            cv2.putText(display_frame, f"Wrist: ({wx},{wy})", (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, x_status, (20, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, x_color, 3)
            cv2.putText(display_frame, z_status, (20, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, z_color, 3)
        else:
            cv2.putText(display_frame, "Not detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        badge_color = (255, 180, 0) if detector_label.startswith("POSE") else (0, 0, 200)
        cv2.rectangle(display_frame, (w - 140, 10), (w - 10, 45), badge_color, -1)
        cv2.putText(display_frame, detector_label, (w - 130, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

        cv2.putText(display_frame, f"{fps:.1f} FPS  [{device.upper()}]",
                    (w - 220, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 255), 2)

        cv2.putText(display_frame, f"Cam Tilt: {camera_tilt_deg:.1f} deg",
                    (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 255), 2)

        # write control instructions
        help_lines = [
            "]: Tilt CW",
            "[: Tilt CCW",
            "C: Switch Cam",
            "R / Peace: Re-lock Person",
            "Z / Thumbs Up: Lock Zoom",
            "ESC: Exit"
        ]
        y0 = h - 20
        for i, text in enumerate(help_lines[::-1]):
            cv2.putText(display_frame, text,
                        (w - 280, y0 - i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        if fade_message:
            current_time = cv2.getTickCount() / cv2.getTickFrequency()
            elapsed = current_time - fade_start_time
            if elapsed < fade_duration:
                alpha = 1.0 - (elapsed / fade_duration)
                overlay = display_frame.copy()
                cv2.putText(overlay, fade_message, (50, h - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                cv2.addWeighted(overlay, alpha, display_frame, 1 - alpha, 0, display_frame)
            else:
                fade_message = ""

        cv2.imshow("Forearm Tracker", display_frame)

        # inputs

        # find closest hand since mediapipe has a depth coordinate unlike yolo
        hand_lms = None
        if results_mp.hand_landmarks:
            best = None
            best_size = -1.0
            for h_lm in results_mp.hand_landmarks:
                wrist_lm = h_lm[WRIST_LM]
                mid_mcp = h_lm[MIDDLE_MCP]
                size = np.hypot(
                    (wrist_lm.x - mid_mcp.x) * w,
                    (wrist_lm.y - mid_mcp.y) * h,
                )
                if size > best_size:
                    best_size = size
                    best = h_lm

            hand_lms = [(int(lm.x * w), int(lm.y * h)) for lm in best]

        # gesture detection
        peace_now = is_peace(hand_lms)
        thumbs_now = is_thumbs_up(hand_lms)

        if peace_now:
            if peace_start is None:
                peace_start = now
        else:
            peace_start = None

        if thumbs_now:
            if thumbs_start is None:
                thumbs_start = now
        else:
            thumbs_start = None

        cooldown_ok = (now - last_gesture_fire) >= GESTURE_COOLDOWN

        if (peace_start is not None
                and (now - peace_start) >= GESTURE_HOLD
                and cooldown_ok):
            locked_id = None
            last_gesture_fire = now
            peace_start = None
            fade_message = "Gesture: Re-locking person"
            fade_start_time = cv2.getTickCount() / cv2.getTickFrequency()
        elif (thumbs_start is not None
                and (now - thumbs_start) >= GESTURE_HOLD
                and cooldown_ok):
            zoom_locked = not zoom_locked
            last_gesture_fire = now
            thumbs_start = None
            fade_message = "Gesture: Zoom locked" if zoom_locked else "Gesture: Zoom unlocked"
            fade_start_time = cv2.getTickCount() / cv2.getTickFrequency()

        # key inputs

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
        elif key == ord('\\'):
            camera_tilt_deg = 0
            fade_message = f"Tilt reset"
            fade_start_time = cv2.getTickCount() / cv2.getTickFrequency()
        elif key in (ord('r'), ord('R')):
            if cooldown_ok and peace_start is None:
                locked_id = None
                last_gesture_fire = now
                fade_message = "Re-locking person"
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
        elif key in (ord('z'), ord('Z')):
            if cooldown_ok and thumbs_start is None:
                zoom_locked = not zoom_locked
                last_gesture_fire = now
                fade_message = "Zoom ratio locked" if zoom_locked else "Zoom ratio unlocked"
                fade_start_time = cv2.getTickCount() / cv2.getTickFrequency()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
