import numpy as np
import math


def r_x(theta):
    """
    Rotate about X axis
    :param theta: angle in radians
    :return: rotation (radians) about X axis
    """
    ct = math.cos(theta)
    st = math.sin(theta)
    rot_x = np.array([[1., 0., 0.],
                      [0, ct, st],
                      [0, -st, ct]])
    return rot_x


def r_y(theta):
    """
    Rotate about Y axis
    :param theta: angle in radians
    :return: rotation (radians) about Y axis
    """
    ct = math.cos(theta)
    st = math.sin(theta)
    rot_y = np.array([[ct, 0, -st],
                      [0., 1., 0.],
                      [st, 0, ct]])
    return rot_y


def r_z(theta):
    """
    Rotate about Z axis
    :param theta: angle in radians
    :return: rotation (radians) about Z axis
    """
    ct = math.cos(theta)
    st = math.sin(theta)
    rot_z = np.array([[ct, st, 0],
                      [-st, ct, 0.],
                      [0., 0., 1.]])
    return rot_z


def euler_to_rot(r):
    """
    rotation matrix based on euler angles

    R_xyz = R_z(GAMMA) * R_y(BETA) * R_x(ALPHA)

    :param r: array containing rotation about each axis
    :return: rotation matrix corresponding to rotation angles
    """
    rx = r_x(r[0])  # R_x(alpha)
    ry = r_y(r[1])  # R_y(beta)
    rz = r_z(r[2])  # R_z(gamma)

    temp = rz.dot(ry)

    return temp.dot(rx)


def quat_to_rot(q):
    """
    rotation matrix based on quaternions
    :param q: array containing quaternions
    :return: rotation matrix corresponding to quaternion values
    """
    # (w, x, y, z) = q
    row_1 = np.array([1-2*(q[2]**2 + q[3]**2)   , 2*(q[1]*q[2] - q[0]*q[3]) , 2*(q[0]*q[2] + q[1]*q[3])])
    row_2 = np.array([2*(q[1]*q[2]+q[0]*q[3])   , 1-2*(q[1]**2 + q[3]**2)   , 2*(q[2]*q[3] - q[0]*q[1])])
    row_3 = np.array([2*(q[1]*q[3] - q[0]*q[2]) , 2*(q[0]*q[1] + q[2]*q[3]) , 1-2*(q[1]**2 + q[2]**2)])

    rotation = np.vstack((row_1, row_2, row_3))

    return rotation


def rot_to_euler(rot):
    """
    Extract euler angles from a rotation matrix
    :param rot: rotation matrix
    :return: array containing rotation angles
    """
    euler = np.zeros(3)

    r00 = rot[0, 0]
    r01 = rot[0, 1]
    r02 = rot[0, 2]
    r10 = rot[1, 0]
    r11 = rot[1, 1]
    r12 = rot[1, 2]
    r20 = rot[2, 0]
    r21 = rot[2, 1]
    r22 = rot[2, 2]

    beta = math.atan2(r20, math.sqrt(r21**2 + r22**2))
    cb = math.cos(beta)
    # if cb < finfo(float).eps:
    #     euler[0] = 10
    #     euler[1] = 10
    #     euler[2] = 10
    #     return euler
    # else:
    gamma = math.atan2(r10/-cb, r00/cb)
    alpha = math.atan2(r21/-cb, r22/cb)
    euler[0] = alpha
    euler[1] = beta
    euler[2] = gamma
    return euler

