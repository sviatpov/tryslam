import numpy as np
import cv2
import os
from multiprocessing import Process, Queue, set_start_method
from FeatureExtractor import FeatureExtractor
from undistort import undistort, K
from Pangolin import vizualizer, draw_process
from reprojection import draw_cross
import os
dir = "/home/sviat/Documents/Datasets/kitti/2011_09_26/2011_09_26_drive_0009_sync/image_00/data/"
# dir = "/home/sviatoslavpovod/Downloads/mav0/cam0/data/"




def draw_lines(im, pts1, pts2):
	for p1, p2 in zip(pts1, pts2):
		im = cv2.line(im, (int(p1[0]), int(p1[1])), \
					  (int(p2[0]), int(p2[1])), color=(0, 255, 0))

def Rt4(R, T):
	RT = np.concatenate((R, T), axis=1)
	RT = np.concatenate((RT, np.array([0, 0, 0, 1]).reshape(1, 4)), axis=0)
	return RT

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
	D4prev = np.array([0, 0, 0]).reshape(3, 1)
	col_prev = np.array([0, 0, 0]).reshape(1, 3)
	q = Queue()

	p = Process(target=draw_process, args=(q,))
	p.start()

	while True:
		for imname in im_list:
			im = cv2.imread(dir + imname)
			# im = undistort(im)
			f2 = feate.extract(im)
			if f1 != 0:
				pts2, pts1 = feate.matches_frame(f1, f2)
				R, T = feate.getRT(pts1, pts2)
				D4 = feate.get3D(R, T, pts1.T, pts2.T).T

				D4, pts1, pts2 = remove_invisible([D4, pts1, pts2])
					# compute and save origins of each frame
					# in first frame coordinate system
				# origin = RTprev @ np.asarray([0, 0, 0, 1])
				# origins.append(origin[:-1] / 10)

				## Draw im
				draw_cross(im, pts2)
				pts1_predicted = (K @ (R @ D4[:, :3].T + T)).T
				pts1_predicted = pts1_predicted[:, :2] / pts1_predicted[:, 2:3]
				draw_cross(im, pts1_predicted, (0, 0, 255))

				D4 = RTprev @ D4.T
				D4 = D4[:-1] / D4[3]

				RT = Rt4(R.T, -R.T @ T)
				RTprev = RTprev @ RT
				D4prev = np.concatenate((D4prev, D4), axis=1)
				cameras.append(RTprev)


				#### DRAW STAFFF
				color = create_color(pts2, im)
				col_prev = np.concatenate((col_prev, color), axis=0)
				# pts1_predicted = (K @ D4.T).T
				# pts1_predicted = pts1_predicted[:, :2] / pts1_predicted[:, 2:3]



				q.put({'points' : D4prev.T, 'colors' : col_prev, "cameras" : cameras})

			f1 = f2
			cv2.imshow("sdc", im)
			cv2.waitKey(0)