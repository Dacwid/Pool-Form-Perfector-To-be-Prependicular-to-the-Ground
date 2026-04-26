"""
=============================================================================
HAND DATA COLLECTION TOOL
=============================================================================
Use this to build your training dataset — photos of your stroking arm
gripping a cue stick, captured from behind.

CAMERA POSITION (fixed for all shots):
  - Behind you, slightly to the side of your stroking arm
  - Height: elbow level
  - Angle: facing forward along your arm toward the target
  - Distance: far enough that ELBOW TO FINGERTIPS are all visible in frame
  - The entire stroking arm must be in frame every shot

LANDMARKS YOU WILL LABEL IN CVAT (11 points per image):
  0  — Elbow
  1  — Wrist
  2  — Index knuckle base
  3  — Middle knuckle base
  4  — Ring knuckle base
  5  — Pinky knuckle base
  6  — Index fingertip
  7  — Middle fingertip
  8  — Ring fingertip
  9  — Pinky fingertip
  10 — Thumb tip

CONTROLS:
  SPACE — save current frame
  N     — next category
  C     — switch camera
  ESC   — exit

TARGET: ~650 images total
=============================================================================
"""

import cv2
import os
import time

SHOT_GUIDE = [
    # -------------------------------------------------------------------------
    # NEUTRAL GRIP
    # Wrist straight, cue inline with forearm, no rotation
    # -------------------------------------------------------------------------
    {
        "name": "neutral_perpendicular",
        "instruction": "NEUTRAL GRIP — forearm perpendicular to ground",
        "detail": "Wrist straight. Forearm pointing straight down. Ideal technique position.",
        "target": 60,
    },
    {
        "name": "neutral_angled_forward",
        "instruction": "NEUTRAL GRIP — forearm angled forward 20-30 deg",
        "detail": "Wrist straight. Cue angled forward as if following through a stroke.",
        "target": 50,
    },
    {
        "name": "neutral_angled_back",
        "instruction": "NEUTRAL GRIP — forearm angled back 20-30 deg",
        "detail": "Wrist straight. Cue angled back as if pulling back for a stroke.",
        "target": 50,
    },

    # -------------------------------------------------------------------------
    # INWARD GRIP (pronated)
    # Wrist rotated inward — knuckles facing more downward
    # -------------------------------------------------------------------------
    {
        "name": "inward_perpendicular",
        "instruction": "INWARD GRIP — forearm perpendicular to ground",
        "detail": "Rotate wrist inward (knuckles facing down). Forearm straight down.",
        "target": 60,
    },
    {
        "name": "inward_angled_forward",
        "instruction": "INWARD GRIP — forearm angled forward 20-30 deg",
        "detail": "Rotate wrist inward. Cue angled forward as if following through.",
        "target": 50,
    },
    {
        "name": "inward_angled_back",
        "instruction": "INWARD GRIP — forearm angled back 20-30 deg",
        "detail": "Rotate wrist inward. Cue angled back as if pulling back.",
        "target": 50,
    },

    # -------------------------------------------------------------------------
    # OUTWARD GRIP (supinated)
    # Wrist rotated outward — knuckles facing more upward
    # -------------------------------------------------------------------------
    {
        "name": "outward_perpendicular",
        "instruction": "OUTWARD GRIP — forearm perpendicular to ground",
        "detail": "Rotate wrist outward (knuckles facing up). Forearm straight down.",
        "target": 60,
    },
    {
        "name": "outward_angled_forward",
        "instruction": "OUTWARD GRIP — forearm angled forward 20-30 deg",
        "detail": "Rotate wrist outward. Cue angled forward as if following through.",
        "target": 50,
    },
    {
        "name": "outward_angled_back",
        "instruction": "OUTWARD GRIP — forearm angled back 20-30 deg",
        "detail": "Rotate wrist outward. Cue angled back as if pulling back.",
        "target": 50,
    },

    # -------------------------------------------------------------------------
    # VARIATION — lighting and background
    # Same grips, different conditions so the model doesn't overfit
    # -------------------------------------------------------------------------
    {
        "name": "varied_lighting_bright",
        "instruction": "BRIGHT LIGHTING — any grip, perpendicular forearm",
        "detail": "Move to your brightest light source. Full arm visible elbow to fingertips.",
        "target": 40,
    },
    {
        "name": "varied_lighting_dim",
        "instruction": "DIM LIGHTING — any grip, perpendicular forearm",
        "detail": "Turn off some lights or close blinds. Full arm visible elbow to fingertips.",
        "target": 40,
    },
    {
        "name": "varied_background",
        "instruction": "DIFFERENT BACKGROUND — any grip, any forearm angle",
        "detail": "Move to a different wall or background. Full arm visible elbow to fingertips.",
        "target": 40,
    },
]


def make_dirs(guide):
    base = os.path.join(os.path.dirname(__file__), "dataset", "raw")
    for shot in guide:
        os.makedirs(os.path.join(base, shot["name"]), exist_ok=True)
    return base


def count_existing(base_dir, category_name):
    path = os.path.join(base_dir, category_name)
    if not os.path.exists(path):
        return 0
    return len([f for f in os.listdir(path) if f.endswith(".jpg")])


def draw_overlay(frame, shot, saved_count, total_saved, flash):
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 140), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # Category instruction
    cv2.putText(frame, shot["instruction"],
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Detail
    cv2.putText(frame, shot["detail"],
                (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)

    # Elbow reminder
    cv2.putText(frame, "CHECK: full arm elbow-to-fingertips in frame",
                (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 200, 255), 1)

    # Progress bar
    target = shot["target"]
    progress = min(saved_count / target, 1.0)
    bar_w = w - 20
    cv2.rectangle(frame, (10, 95), (10 + bar_w, 112), (60, 60, 60), -1)
    fill_color = (0, 200, 0) if progress < 1.0 else (0, 255, 100)
    cv2.rectangle(frame, (10, 95), (10 + int(bar_w * progress), 112), fill_color, -1)
    cv2.putText(frame, f"{saved_count}/{target} photos",
                (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    # Total
    cv2.putText(frame, f"Total: {total_saved}",
                (w - 130, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

    # Flash
    if flash:
        bright = frame.copy()
        cv2.rectangle(bright, (0, 0), (w, h), (255, 255, 255), -1)
        cv2.addWeighted(bright, 0.4, frame, 0.6, 0, frame)

    # Controls bar at bottom
    cv2.rectangle(frame, (0, h - 30), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, "SPACE: Save    N: Next category    C: Switch cam    ESC: Exit",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    # Complete banner
    if saved_count >= target:
        cv2.putText(frame, "DONE! Press N for next category.",
                    (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)


def main():
    base_dir = make_dirs(SHOT_GUIDE)

    camera_index = 0
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera.")

    shot_index = 0
    flash_until = 0

    total_target = sum(s["target"] for s in SHOT_GUIDE)

    print("\n  Hand Data Collection Tool")
    print("  ==========================")
    print(f"  Saving to: {base_dir}")
    print(f"  {len(SHOT_GUIDE)} categories | Target: {total_target} total images")
    print()
    print("  CAMERA REMINDER:")
    print("  - Position behind your stroking arm, at elbow height")
    print("  - Elbow to fingertips must be fully visible every shot")
    print()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        shot = SHOT_GUIDE[shot_index]
        category_dir = os.path.join(base_dir, shot["name"])
        saved_count = count_existing(base_dir, shot["name"])
        total_saved = sum(count_existing(base_dir, s["name"]) for s in SHOT_GUIDE)
        flash = time.time() < flash_until

        draw_overlay(frame, shot, saved_count, total_saved, flash)
        cv2.imshow("Data Collection", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break

        elif key == ord(' '):
            timestamp = int(time.time() * 1000)
            filename = os.path.join(category_dir, f"{timestamp}.jpg")
            cv2.imwrite(filename, frame)
            flash_until = time.time() + 0.1
            print(f"  Saved: {shot['name']}/{timestamp}.jpg  ({saved_count + 1}/{shot['target']})")

        elif key in (ord('n'), ord('N')):
            shot_index = (shot_index + 1) % len(SHOT_GUIDE)
            print(f"\n  → {SHOT_GUIDE[shot_index]['name']}")

        elif key in (ord('c'), ord('C')):
            new_index = 1 - camera_index
            new_cap = cv2.VideoCapture(new_index)
            if new_cap.isOpened():
                cap.release()
                cap = new_cap
                camera_index = new_index
                print(f"  Switched to camera {camera_index}")
            else:
                new_cap.release()
                print("  No other camera found")

    cap.release()
    cv2.destroyAllWindows()

    print("\n  Collection complete. Summary:")
    print("  ─────────────────────────────")
    total = 0
    for shot in SHOT_GUIDE:
        count = count_existing(base_dir, shot["name"])
        status = "✓" if count >= shot["target"] else f"{count}/{shot['target']}"
        print(f"    {shot['name']:<35} {status}")
        total += count
    print(f"\n  Total images collected: {total} / {total_target}")
    print(f"  Saved to: {base_dir}")


if __name__ == "__main__":
    main()
