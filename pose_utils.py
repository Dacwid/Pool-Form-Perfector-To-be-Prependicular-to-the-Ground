import numpy as np
import math

def vector(p1, p2) :
    #finds distance of two coordinates
    return np.array([p2[0] - p1[0], p2[1] - p1[1]])

def angle_between(v1, v2) :
    #applies the formula to determine the angle based on v1, vector between elbow and wrist, and v2, gravity vector
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return  math.degrees(math.acos(dot))

def wrist_in_line(wrist_x, wrist_z, frame_width, x_threshold = 0.1, z_threshold = 0.15 ) :
    #makes sure the wrist is in line with the camera for proper detection
    #threshold makes sure wrist doesn't have to be in the perfect right place but still in the center of the screen
    #and not too far away from the camera
    center_x = frame_width / 2
    delta_x = abs(wrist_x - center_x)

    horizontal_ok = delta_x <= x_threshold * frame_width
    depth_ok = wrist_z < z_threshold

    return horizontal_ok and depth_ok

def get_pixel_coordinates(landmark, frame_width, frame_height) :
    x = int(landmark.x * frame_width)
    y = int(landmark.y * frame_height)
    z = landmark.z
    return x, y, z

def forearm_angle(elbow, wrist, gravity_vec) :
    arm_vec = vector(elbow, wrist)
    return angle_between(arm_vec, gravity_vec)