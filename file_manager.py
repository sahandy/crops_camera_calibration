import csv
import sys
import numpy as np

__author__ = 'sahand'

point_x = []
point_y = []
point_z = []
quat_x = []
quat_y = []
quat_z = []
quat_w = []


"""
Parse marker pose info
"""
# reading marker pose information from file
f = open("calibration_data/marker_pose.csv")
# find file from command-line argument
# f = open(sys.argv[1], 'rt')
csv_f = csv.reader(f)
for row in csv_f:
    if csv_f.line_num == 1:
        continue
    point_x.append(row[0])
    point_y.append(row[1])
    point_z.append(row[2])
    quat_x.append(row[3])
    quat_y.append(row[4])
    quat_z.append(row[5])
    quat_w.append(row[6])

pose_t = np.vstack((point_x, point_y, point_z))
pose_r = np.vstack((quat_w, quat_x, quat_y, quat_z))

"""
Parse snake pose info
"""
snake_x = []
snake_y = []
snake_z = []
snake_alpha = []
snake_beta = []
snake_gamma = []

f = open("calibration_data/snake_pose.csv")
csv_f = csv.reader(f)
for row in csv_f:
    if csv_f.line_num == 1:
        continue
    snake_x.append(row[0])
    snake_y.append(row[1])
    snake_z.append(row[2])
    snake_alpha.append(row[3])
    snake_beta.append(row[4])
    snake_gamma.append(row[5])

snake_t = np.vstack((snake_x, snake_y, snake_z))
snake_r = np.vstack((snake_alpha, snake_beta, snake_gamma))
