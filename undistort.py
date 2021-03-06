import numpy as np
import cv2

  ### Camera matrix from calib file

  #	fx 0 cx
  #	0 fy cy
  # 0 0   1

K = np.array([[984.2439,   0,     690.0000],
			  [0,      980.8141,  233.1966],
			  [0,         0,           1]])

distort_param = np.array([-3.728755e-01, 2.037299e-01, 2.219027e-03, 1.383707e-03, -7.233722e-02])
# K = np.array([[458.654,   0,     367.215],
#         [0,      457.296,  248.375],
#         [0,         0,           1]])
#
# distort_param = np.array([-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05])

def undistort(im):
    dst = cv2.undistort(im, K, distort_param)
    return dst