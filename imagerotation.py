import matplotlib.pyplot as plt
import numpy as np
from numpy.lib import angle

from sparsecoordinatematrices import image, apply_transform, homography


def transform_rotate_about_center(shape, degrees):
    c = np.cos(np.deg2rad(angle))
    s = np.sin(np.deg2rad(angle))

    H_rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    center = np.array(image.shape) / 2
    H_tr0 = np.array([[1, 0, -center[0]],
                      [0, 1, -center[1]],
                      [0, 0, 1]])
    H_tr1 = np.array([[1, 0, center[0]],
                      [0, 1, center[1]],
                      [0, 0, 1]])

    H_rot_cent = H_tr1 @ H_rot @ H_tr0
    sparse_op = homography(H_rot_cent,image.shape)
    return sparse_op

tf = transform_rotate_about_center(image.shape, 30)
plt.imshow(apply_transform(image, tf))