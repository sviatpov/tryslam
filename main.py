import numpy as np
import cv2
import os
from multiprocessing import Process, Queue, set_start_method
from FeatureExtractor import FeatureExtractor
from undistort import undistort, K

from Pangolin import vizualizer, draw_process
from map import Frame, Map, Point
from reprojection import draw_cross

import os
dir = "/home/sviat/Documents/Datasets/kitti/2011_09_26/2011_09_26_drive_0009_sync/image_00/data/"


def draw_lines(im, pts1, pts2):
	for p1, p2 in zip(pts1, pts2):
		im = cv2.line(im, (int(p1[0]), int(p1[1])), \
					  (int(p2[0]), int(p2[1])), color=(0, 255, 0))

def remove_invisible(args):

	args[0] = args[0][:, :4] / args[0][:, 3:4]
	mask = args[0][:, 2] > 0.05
	ret = []
	for e in args:
		assert e.shape[0] == mask.shape[0]
		ret.append(e[mask])

	return ret
	# for i in range(len(args)):
	# 	args[i] = args[i][mask]
def create_color(pts2, im):
	pts2 = np.asarray(pts2, dtype='uint8')
	color = im[pts2[:, 0], pts2[:, 1]] / 255.
	return color



if __name__ == "__main__":

	im_list = sorted(os.listdir(dir))
	feate = FeatureExtractor(K)
	f1 = 0
	RTprev = np.eye(4)
	origins = []
	cameras = []
	D4prev = np.array([0, 0, 0, 0]).reshape(4, 1)
	col_prev = np.array([0, 0, 0]).reshape(1, 3)

	mapa = Map()

	### Vizualization process
	q = Queue()
	p = Process(target=draw_process, args=(q,))
	p.start()

	### INIT FIRST FRAME instead of check of first frame
	im = cv2.imread(dir + im_list[0])
	Fr = Frame(im)
	Fr.pose = np.eye(4)
	Fr.add_feature(feate.extract(Fr.im))
	mapa.append_frame(Fr)
	cameras.append(np.eye(4))

	##### Process another frame

	for it in range(1, len(im_list)):
		im = cv2.imread(dir + im_list[it])
		Fr = Frame(im)

		Fr.add_feature(feate.extract(Fr.im))

		mapa.append_frame(Fr)

		idx1, idx2, D4, pts2 = feate.estimate_pose(*(mapa.get_two_back_frame()),
												   save_pose=True)

		[mapa.append_point(Point(p)) for p in D4]

		D4 = mapa.frames[-2].pose @ D4.T

		D4prev = np.concatenate((D4prev, D4), axis=1)

		cameras.append(mapa.frames[-2].pose)

		color = create_color(pts2, im)

		col_prev = np.concatenate((col_prev, color), axis=0)

	# if f1 != 0:
		# 		# compute and save origins of each frame
		# 		# in first frame coordinate system
		# 	# origin = RTprev @ np.asarray([0, 0, 0, 1])
		# 	# origins.append(origin[:-1] / 10)
		#
		# 	## Draw im
		# 	draw_cross(im, pts2)
		# 	pts1_predicted = (K @ (R @ D4[:, :3].T + T)).T
		# 	pts1_predicted = pts1_predicted[:, :2] / pts1_predicted[:, 2:3]
		# 	draw_cross(im, pts1_predicted, (0, 0, 255))
		#
		# 	D4 = RTprev @ D4.T
		# 	D4 = D4[:-1] / D4[3]
		#
		# 	RT = Rt4(R.T, -R.T @ T)
		# 	RTprev = RTprev @ RT
		# 	D4prev = np.concatenate((D4prev, D4), axis=1)
		# 	cameras.append(RTprev)
		#
		#
		# 	#### DRAW STAFFF
		# 	color = create_color(pts2, im)
		# 	col_prev = np.concatenate((col_prev, color), axis=0)
		# 	# pts1_predicted = (K @ D4.T).T
		# 	# pts1_predicted = pts1_predicted[:, :2] / pts1_predicted[:, 2:3]



		q.put({'points' : D4prev.T, "cameras" : cameras, "colors" : col_prev})

		cv2.imshow("sdc", im)
		cv2.waitKey(0)