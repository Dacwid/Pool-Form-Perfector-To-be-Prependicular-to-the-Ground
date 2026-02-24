import numpy as np

def get_gravity_vector(accelerometer_data=None) :
    """Returns the 2d gravity vector of the source based on accelerometer(e.g. np.array([gx, gy]))"""
    if accelerometer_data is not None :
        ax, ay, az = accelerometer_data
        gx = ax
        gy = ay
        gravity_vec = np.array([gx, gy])
        norm = np.linalg.norm(gravity_vec)
        if norm == 0 :
            return np.array([0,1])
        return gravity_vec / norm
    else :
        return np.array([0, 1])