import numpy as np
import cv2
import os
from FeatureExtractor import FeatureExtractor
from undistort import undistort, K
from Pangolin import vizualizer
# dir = "/home/sviatoslavpovod/Documents//datasets/2011_09_26/2011_09_26_drive_0036_sync/image_00/data/"
# dir = "/home/sviatoslavpovod/Downloads/mav0/cam0/data/"



im_list = sorted(os.listdir(dir))
feate = FeatureExtractor(K)
f1 = 0
v = vizualizer()
RTprev = np.eye(4)
origins = []
D4prev = np.array([0,0,0]).reshape(3,1)

def draw_lines(im, pts1, pts2):
	for p1, p2 in zip(pts1, pts2):
		im = cv2.line(im, (int(p1[0]), int(p1[1])), \
					  (int(p2[0]), int(p2[1])), color=(0, 255, 0))

def Rt4(R, T):
	RT = np.concatenate((R, T), axis=1)
	RT = np.concatenate((RT, np.array([0, 0, 0, 1]).reshape(1, 4)), axis=0)
	return RT

while True:
	for imname in im_list:
		im = cv2.imread(dir + imname)
		im = undistort(im)
		f2 = feate.extract(im)
		if f1 != 0:
			pts2, pts1 = feate.matches_frame(f1, f2)
			R, T = feate.getRT(pts1, pts2)
			D4 = feate.get3D(R, T, pts1.T, pts2.T).T

				## remove not vizible points
			D4 = D4[:, :4] / D4[:, 3:4]
			mask = D4[:, 2] > 0.05
			D4 = D4[mask]

				# compute and save origins of each frame
				# in first frame coordinate system
			origin = RTprev @ np.asarray([0,0,0,1])
			origins.append(origin[:-1] / 10)


			D4 = RTprev @ D4.T
			D4 = D4[:-1] / D4[3]

			RT = Rt4(R.T, -R.T @ T)
			RTprev = RTprev @ RT
			print(RTprev[:, 3])
			D4prev = np.concatenate((D4prev, D4), axis=1)

			v.main(origins, D4prev.T / 10)
		f1 = f2
		cv2.imshow("sdc", im)
		cv2.waitKey(1)