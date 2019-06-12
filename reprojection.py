import numpy as np
import cv2
import os
from FeatureExtractor import FeatureExtractor
from undistort import undistort, K
from Pangolin import vizualizer

import sys



def draw_lines(im, pts1, pts2):
    for p1, p2 in zip(pts1, pts2):
        im = cv2.line(im, (int(p1[0]), int(p1[1])), \
        (int(p2[0]), int(p2[1])), color=(0, 255, 0))

def Rt4(R, T):
    RT = np.concatenate((R, T), axis=1)
    RT = np.concatenate((RT, np.array([0, 0, 0, 1]).reshape(1, 4)), axis=0)
    return RT

def draw_cross(im, pts, color=(255, 0, 0)):
    for p in pts:
        cv2.drawMarker(im, (int(p[0]), int(p[1])), color, cv2.MARKER_CROSS, markerSize=5)

def draw_kpts_cross(im, kpts, color=(255, 0, 0)):
    for p in kpts:
        cv2.drawMarker(im, (int(p.pt[0]), int(p.pt[1])), color, cv2.MARKER_CROSS, markerSize=5)

if __name__ == "__main__":

    dir = "/home/sviat/Documents/Datasets/kitti/2011_09_26/2011_09_26_drive_0009_sync/image_00/data/"
    # dir = "/home/sviatoslavpovod/Downloads/mav0/cam0/data/"

    im_list = sorted(os.listdir(dir))
    feate = FeatureExtractor(K)
    f1 = 0
    RTprev = np.eye(4)
    origins = []
    D4prev = np.array([0, 0, 0]).reshape(3, 1)
    im_prev = 0
    while True:
        for imname in im_list:
            im = cv2.imread(dir + imname)
            f2 = feate.extract(im)

            if f1 != 0:

                ## 2D points on image
                pts2, pts1 = feate.matches_frame(f1, f2)

                ## R, T between cameras
                R, T = feate.getRT(pts1, pts2)

                ## 3D points
                D4 = feate.get3D(R, T, pts1.T, pts2.T).T
                D4 = D4[:, :3] / D4[:, 3:4]
                mask = D4[:, 2] > 0.05

                D4 = D4[mask]
                pts1 = pts1[mask]

                pts1_predicted = (K @ R @ (D4.T + T)).T
                pts1_predicted = pts1_predicted[:, :2] / pts1_predicted[:, 2:3]

                draw_cross(im, pts2)

                draw_cross(im, pts1_predicted, (0, 0, 255))

                cv2.imshow('kpts', im_prev)
                cv2.waitKey(0)
            f1 = f2
            im_prev = im

# while True:
# 	for imname in im_list:
# 		im = cv2.imread(dir + imname)
# 		# im = undistort(im)
# 		f2 = feate.extract(im)
# 		if f1 != 0:
# 			pts2, pts1 = feate.matches_frame(f1, f2)
# 			R, T = feate.getRT(pts1, pts2)
# 			D4 = feate.get3D(R, T, pts1.T, pts2.T).T
#
# 				## remove not vizible points
# 			D4 = D4[:, :4] / D4[:, 3:4]
# 			mask = D4[:, 2] > 0.05
# 			D4 = D4[mask]
#
# 				# compute and save origins of each frame
# 				# in first frame coordinate system
# 			origin = RTprev @ np.asarray([0,0,0,1])
# 			origins.append(origin[:-1] / 10)
#
#
# 			D4 = RTprev @ D4.T
# 			D4 = D4[:-1] / D4[3]
#
# 			RT = Rt4(R.T, -R.T @ T)
# 			RTprev = RTprev @ RT
# 			print(RTprev[:, 3])
# 			D4prev = np.concatenate((D4prev, D4), axis=1)
#
# 			draw_lines(im, pts1, pts2)
# 			# v.main(origins, D4prev.T / 10)
# 		f1 = f2
# 		cv2.imshow("sdc", im)
# 		cv2.waitKey(10)