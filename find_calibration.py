import numpy as np
from numpy.linalg import inv
from numpy.linalg import norm
from scipy.optimize import minimize
import transform as tf
import math


__author__ = 'Sahand Yousefpour'


def calibration_r(R_I_C):
    """

    :param R_I_C: array consists of euler angles (alpha, beta, gamma)
    :return: sum of the norm of angles of all observations
    """
    r_i_c = tf.euler_to_rot(R_I_C)
    r_w_i = tf.euler_to_rot(ideal_cam_r[:, 0])
    residual = 0
    for i in range(0, 8):

        r_w_tcp = tf.euler_to_rot(snake_r[:, i])
        angles = ([1.5707963267948966, 0, -1.5707963267948966])
        r_tcp_m = tf.euler_to_rot(angles)

        r_w_m = r_tcp_m.dot(r_w_tcp)
        r_c_m = tf.quat_to_rot(pose_r[:, i])
        rhs = r_c_m.dot(r_i_c).dot(r_w_i)
        res = inv(r_w_m).dot(rhs)
        angles = tf.rot_to_euler(res)

        residual += norm(angles)

    return residual


def calibration_t(T_I_C):
    """

    :param T_I_C:
    :return:
    """

    # print T_I_C
    homog = ([0, 0, 0, 1])

    # w -> i
    r_w_i = tf.euler_to_rot(ideal_cam_r[:, 0])
    t_w_i = -r_w_i.dot(ideal_cam_t[:, 0])
    A_w_i = np.c_[r_w_i, t_w_i]
    A_w_i = np.vstack((A_w_i, homog))

    # i -> c
    r_i_c = tf.euler_to_rot(result_rotation)
    A_i_c = np.c_[r_i_c, T_I_C]
    A_i_c = np.vstack((A_i_c, homog))

    residual = 0

    for i in range(0, 8):
        # w -> tcp
        r_w_tcp = tf.euler_to_rot(snake_r[:, i])
        # translation vector expressed in TCP coordinate system
        t_w_tcp = -r_w_tcp.dot(snake_t[:, i])
        A_w_tcp = np.c_[r_w_tcp, t_w_tcp]
        A_w_tcp = np.vstack((A_w_tcp, homog))
        # tcp -> m
        angles = ([1.5707963267948966, 0, -1.5707963267948966])
        r_tcp_m = tf.euler_to_rot(angles)
        t_tcp_m = np.zeros(3)
        A_tcp_m = np.c_[r_tcp_m, t_tcp_m]
        A_tcp_m = np.vstack((A_tcp_m, homog))
        # c -> m
        r_c_m = tf.quat_to_rot(pose_r[:, i])
        t_c_m = -r_c_m.dot(pose_t[:, i])
        A_c_m = np.c_[r_c_m, t_c_m]
        A_c_m = np.vstack((A_c_m, homog))
        # .....
        A_w_m_lhs = A_tcp_m.dot(A_w_tcp)
        A_w_m_rhs = A_c_m.dot(A_i_c).dot(A_w_i)
        t_res = inv(A_w_m_lhs).dot(A_w_m_rhs)
        residual += norm(t_res[:, 3])

    return residual


# T_C->M
pose_x = np.array([-0.167018, -0.117128, 0.0645416, 0.0571499, 0.236498, 0.149031, 0.155234, 0.333421])
pose_y = np.array([0.164943, -0.139215, -0.139426, 0.166812, 0.166689, 0.166657, -0.13865, -0.137651])
pose_z = np.array([0.820942, 0.739023, 0.837633, 0.826066, 0.919171, 1.10776, 1.11393, 1.20022])
pose_t = np.vstack((pose_x, pose_y, pose_z))

pose_rx = np.array([0.999708, 0.999567, 0.999717, 0.999953, 0.999937, 0.999772, 0.999966, 0.99997])
pose_ry = np.array([0.00607845, 0.0027874, -0.00110043, 0.00679859, 0.00608006, 0.00737778, 0.00123076, -0.000402028])
pose_rz = np.array([-0.0231964, -0.0279208, -0.0219129, 0.00651526, 0.00901205, 0.0177755, 0.00578322, 0.00708955])
pose_rw = np.array([0.00284671, -0.00881726, -0.0092443, 0.00235054, 0.0026576, 0.00924844, -0.00576046, 0.00323431])
pose_r = np.vstack((pose_rw, pose_rx, pose_ry, pose_rz))

# T_W->M (snake poses)
snake_x = np.array([0.1, 0.4, 0.4, 0.1, 0.1, 0.1, 0.4, 0.4])
snake_y = np.array([0.78, 0.68, 0.68, 0.68, 0.68, 0.88, 0.88, 0.88])
snake_z = np.array([1.37, 1.37, 1.17, 1.17, 0.97, 0.97, 0.97, 0.77])
snake_t = np.vstack((snake_x, snake_y, snake_z))

snake_ra = -0.52 * np.ones(8)
snake_rb = np.zeros(8)
snake_rg = np.zeros(8)
snake_r = np.vstack((snake_ra, snake_rb, snake_rg))

# T_W->I (from world to ideal camera: measured)
# x     : 0.270     meters
# y     : 0.000     meters
# z     : 1.574     meters
# alpha : -120      degrees
# beta  : 0         degree
# gamma : -90       degrees
ideal_cam_x = 0.27 * np.ones(8)
ideal_cam_y = np.zeros(8)
ideal_cam_z = 1.574 * np.ones(8)
ideal_cam_t = np.vstack((ideal_cam_x, ideal_cam_y, ideal_cam_z))

ideal_cam_ra = math.radians(-120) * np.ones(8)
ideal_cam_rb = np.zeros(8)
ideal_cam_rg = math.radians(90) * np.ones(8)
ideal_cam_r = np.vstack((ideal_cam_ra, ideal_cam_rb, ideal_cam_rg))

# MAIN
# Rotation
initial_guess = np.array([0, 0, 0])
res = minimize(calibration_r, initial_guess, method='CG')
result_rotation = np.array(res.x)

# Translation
initial_guess = np.array([0, 0, 0])
res = minimize(calibration_t, initial_guess, method='CG')
result_translation = np.array(res.x)

# average_norm = calibration_t(result_translation)/8 - 1
# sd = math.sqrt(average_norm)
#
# print sd
#
print "======= RES_ROTATION ======="
print(result_rotation), " , in radians"
print "======= RES_TRANSLATION ======="
print(result_translation), ", in meters"

# f = open('calibration_data/result.txt', 'w')
# f.write('RES_ROTATION\n')
# f.write(str(result_rotation[0]))
# f.write('\n')
# f.write(str(result_rotation[1]))
# f.write('\n')
# f.write(str(result_rotation[2]))
# f.write('\n')
# f.write('RES_TRANSLATION\n')
# f.write(str(result_translation[0]))
# f.write(str(result_translation[1]))
# f.write(str(result_translation[2]))
# f.close()

