import numpy as np
import math

def vector(p1, p2) :
    # Finds distance of two coordinates
    return np.array([p1[0] - p2[0], p1[1] - p2[1]])

def angle_between(v1, v2) :
    # Applies the formula to determine the angle based on v1, vector between elbow and wrist, and v2, gravity vector
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return  math.degrees(math.acos(dot))

def wrist_in_line(wrist_x, wrist_z, frame_width, x_threshold = 0.04, z_threshold = 0.15 ) :
    # Makes sure the wrist is in line with the camera for proper detection
    # Threshold makes sure wrist doesn't have to be in the perfect right place but still in the center of the screen
    # and not too far away from the camera
    center_x = frame_width / 2 
    delta_x = abs(wrist_x - center_x)

    horizontal_ok = delta_x <= x_threshold * frame_width
    depth_ok = wrist_z < z_threshold

    return horizontal_ok, depth_ok

def get_pixel_coordinates(landmark, frame_width, frame_height) :
    x = int(landmark.x * frame_width)
    y = int(landmark.y * frame_height)
    z = landmark.z
    return x, y, z

def forearm_angle(wrist, finger, gravity_vec) :
    arm_vec = vector(wrist, finger)
    return angle_between(arm_vec, gravity_vec)

WRIST_LM = 0
THUMB_MCP = 2
THUMB_TIP = 4
INDEX_PIP = 6
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_TIP = 12
RING_PIP = 14
RING_TIP = 16
PINKY_PIP = 18
PINKY_TIP = 20

# basic logic is if tip more than base then that finger is raised, if not then it is down
def is_peace(lms):
    if lms is None:
        return False
    index_extended  = lms[INDEX_TIP][1]  < lms[INDEX_PIP][1]
    middle_extended = lms[MIDDLE_TIP][1] < lms[MIDDLE_PIP][1]
    ring_folded     = lms[RING_TIP][1]   > lms[RING_PIP][1]
    pinky_folded    = lms[PINKY_TIP][1]  > lms[PINKY_PIP][1]
    return index_extended and middle_extended and ring_folded and pinky_folded

def is_thumbs_up(lms):
    if lms is None:
        return False
    thumb_up      = lms[THUMB_TIP][1]  < lms[THUMB_MCP][1]
    index_folded  = lms[INDEX_TIP][1]  > lms[INDEX_PIP][1]
    middle_folded = lms[MIDDLE_TIP][1] > lms[MIDDLE_PIP][1]
    ring_folded   = lms[RING_TIP][1]   > lms[RING_PIP][1]
    pinky_folded  = lms[PINKY_TIP][1]  > lms[PINKY_PIP][1]
    return thumb_up and index_folded and middle_folded and ring_folded and pinky_folded