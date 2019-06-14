import cv2
import numpy as np
# import g2o






def Rt4(R, T):
	RT = np.concatenate((R, T), axis=1)
	RT = np.concatenate((RT, np.array([0, 0, 0, 1]).reshape(1, 4)), axis=0)
	return RT


def Rt4_inv(R, T):
	return Rt4(R.T, -R.T @ T)


def inv_orto_Rt4(RT):
	Rinv = RT[:3, :3].T
	T = - Rinv @ RT[:3, 3]
	RT[:3, :3] = Rinv
	RT[:3, 3] = T


class FeatureExtractor():

	def __init__(self, K = 0):
		self.orb = cv2.ORB_create()
		self.mtch = cv2.BFMatcher(cv2.HAMMING_NORM_TYPE)
		self.F = 0
		self.K = K

	def getRT(self, pts1, pts2):

		E = self.K.T @ self.F @ self.K
		_, R, T, mask = cv2.recoverPose(E, pts1, pts2, self.K)

		### TODO : test

		pts1 = pts1[mask[:,0] == 1]
		pts2 = pts2[mask[:,0] == 1]
		return R, T

	def get3D(self, R, T, pts1, pts2):

		# Indentity projection matrix
		I = np.concatenate((np.eye(3, dtype='float32'), np.array([[0],[0],[0]])), axis=1) # ------> I,shape = [3,4].

		# Proj
		RT = np.concatenate((R, T), axis=1) 											# -----> R.shape = [3,3],
																						# -----> T.shape = [3,1],
																						# -----> RT.shape = [3,4];
		return cv2.triangulatePoints(self.K @ I,self.K @ RT, pts1, pts2)

	def extract(self, im):
		""" Extract Features and compute ORB descriptors from image

		@:param im - image
		@:returns tuple of arrays - (keypoints, descriptirs)

		"""

		# 1. extract  features
		# TODO is needed use goodFeature  or just init ORB with def number of features
		feats = cv2.goodFeaturesToTrack(np.mean(im, axis=2).astype(np.uint8), \
										5000, \
										qualityLevel=0.01, \
										minDistance=3) # Shape  ---> [nfeature, 1, im]

		# 2. compute descriptors from detected features
		kpts = [cv2.KeyPoint(f[0][0], f[0][1], _size=20) for f in feats]
		kpts, des = self.orb.compute(im, kpts, None)

		return kpts, des

	def matches_frame(self, f1, f2):

		""" Compute matches of input descriptors

		@:param f1 - tuple of (keypoints, descriptors) for image1
		@:param f2 - the same for im2
		@:return tuple of filtered correspondent points

		"""
		kpts, des = f2.key_pts, f2.des
		source_kpts, source_des = f1.key_pts, f1.des
		pts1, pts2 = [], []
		idx1, idx2 = [], []
		idx1s, idx2s = set(), set()

		matches = self.mtch.knnMatch(source_des, des, k=2)

		# FixMe check are we need this strange filter for matches
		matches = sorted(matches, key=lambda x:x[0].distance)
		for m, n in matches:
			if m.distance < 0.7 * n.distance:
				# good.append(m)
				if m.distance < 10:
					if m.queryIdx not in idx1s and m.trainIdx not in idx2s:
						idx1.append(m.queryIdx)
						idx2.append(m.trainIdx)
						idx1s.add(m.queryIdx)
						idx2s.add(m.trainIdx)
						pts2.append(kpts[m.trainIdx].pt)
						pts1.append(source_kpts[m.queryIdx].pt)


		## Why pts should be float ????????
		pts1, pts2 = np.asarray(pts1, dtype='float32'), np.asarray(pts2, dtype='float32')
		idx1, idx2 = np.asarray(idx1, dtype='uint8'), np.asarray(idx2, dtype='uint8')
		self.F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)

		idx1 = idx1[mask[:, 0] == 1]
		idx2 = idx2[mask[:, 0] == 1]
		pts1 = pts1[mask[:, 0] == 1]
		pts2 = pts2[mask[:, 0] == 1]

		return idx1, pts1,  idx2, pts2


	def filter_invisible(self, *args, mask):

		ret = []
		for e in args:
			ret.append(e[mask])
		return ret

	def estimate_pose(self, f1, f2, save_pose = True):

		## match features on two images

		idx1, pts1, idx2, pts2 = self.matches_frame(f1, f2)

		## estimate transformation from matched pixels
		R, T = self.getRT(pts1, pts2)

		## save pose into
		if save_pose:
			f2.pose = f1.pose @ Rt4_inv(R, T)

		D4 = self.get3D(R, T, pts1.T, pts2.T)

		## TODO check do we need it
		D4 = D4.T
		D4 = D4 / D4[:,-1, np.newaxis]
		D4, idx1, idx2, pts2 = self.filter_invisible(D4, idx1, idx2,pts2, mask=D4[:, -2] > 0.05)

		return idx1, idx2, D4, pts2


