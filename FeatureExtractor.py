import cv2
import numpy as np
# import g2o

class FeatureExtractor():
	def __init__(self, K=0):
		self.orb = cv2.ORB_create()
		self.mtch = cv2.BFMatcher(cv2.HAMMING_NORM_TYPE)

		self.F = 0
		self.K = K

	def getRT(self, pts1, pts2):

		E = self.K.T @ self.F @ self.K
		_, R, T, mask = cv2.recoverPose(E, pts1, pts2, self.K)
		pts1 = pts1[mask[:,0] == 1]
		pts2 = pts2[mask[:,0] == 1]
		# print(R)
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
		kpts, des = f2[0], f2[1]
		source_kpts, source_des = f1[0], f1[1]
		pts1, pts2 = [], []
		idx1, idx2 = [], []
		idx1s, idx2s = set(), set()

		matches = self.mtch.knnMatch(source_des, des, k=2)

		# FixMe check are we need this strange filter for matches
		matches = sorted(matches, key=lambda x:x[0].distance)
		for m, n in matches:
			if m.distance < 0.7 * n.distance:
				# good.append(m)
				if m.distance < 32:
					if m.queryIdx not in idx1s and m.trainIdx not in idx2s:
						idx1.append(m.queryIdx)
						idx2.append(m.trainIdx)
						idx1s.add(m.queryIdx)
						idx2s.add(m.trainIdx)
						pts2.append(kpts[m.trainIdx].pt)
						pts1.append(source_kpts[m.queryIdx].pt)


		pts1, pts2 = np.asarray(pts1, dtype='float32'), np.asarray(pts2, dtype='float32')
		self.F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
		pts1 = pts1[mask[:, 0] == 1]
		pts2 = pts2[mask[:, 0] == 1]

		return pts2, pts1

class Fundamental():
	def __init__(self):
		pass
	def get_fundamental(self, pts1, pts2):
		F = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)
		return F

class Frame():
	def __init__(self, img):
		self.pts, self.des = FeatureExtractor().extract(img)
		pass