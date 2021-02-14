import numpy as np


def fast_normalise(q):
    """
        https://github.com/KieranWynn/pyquaternion/blob/99025c17bab1c55265d61add13375433b35251af/pyquaternion/quaternion.py#L513
        Normalise the object to a unit quaternion using a fast approximation method if appropriate.
        Object is guaranteed to be a quaternion of approximately unit length
        after calling this operation UNLESS the object is equivalent to Quaternion(0)
        """
    
    mag_squared = np.dot(q, q)
    if (mag_squared == 0):
        return
    if (abs(1.0 - mag_squared) < 2.107342e-08):
        mag =  ((1.0 + mag_squared) / 2.0) # More efficient. Pade approximation valid if error is small
    else:
        mag =  np.sqrt(mag_squared) # Error is too big, take the performance hit to calculate the square root properly
        
    return q / mag


def quaternion_rotation_matrix(Q):
    """
        https://automaticaddison.com/how-to-convert-a-quaternion-to-a-rotation-matrix/
        Covert a quaternion into a full three-dimensional rotation matrix.
        
        Input
        :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3) 
        
        Output
        :return: A 3x3 element matrix representing the full 3D rotation matrix. 
             This rotation matrix converts a point in the local reference 
             frame to a point in the global reference frame.
        """
    # Extract the values from Q
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]
    
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
    
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
    
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
    
    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]]) 
    return rot_matrix


def from_axis_angle(axis, angle):
    """Initialise from axis and angle representation
    Create a Quaternion by specifying the 3-vector rotation axis and rotation
    angle (in radians) from which the quaternion's rotation should be created.
    Params:
        axis: a valid numpy 3-vector
        angle: a real valued angle in radians
    """
    mag_sq = np.dot(axis, axis)
    if mag_sq == 0.0:
        raise ZeroDivisionError("Provided rotation axis has no length")
    # Ensure axis is in unit vector form
    if (abs(1.0 - mag_sq) > 1e-12):
        axis = axis / np.sqrt(mag_sq)
    theta = angle / 2.0
    r = np.cos(theta)
    i = axis * np.sin(theta)

    return np.array([r, i[0], i[1], i[2]])
    
