import cv2
import numpy as np

class FeatureExtractor():
	def __init__(self):
		self.orb = cv2.ORB_create()
		self.mtch = cv2.BFMatcher(cv2.HAMMING_NORM_TYPE)

		self.sourse_des = None
		self.sourse_kpts = None

	def extract(self, im):

		# detect features

		feats = cv2.goodFeaturesToTrack(np.mean(im, axis=2).astype(np.uint8), \
										3000, \
										qualityLevel=0.01, \
										minDistance=3)

		# compute descriptors from detected features

		kpts = [cv2.KeyPoint(f[0][0], f[0][1], _size=20) for f in feats]
		kpts, des = self.orb.compute(im, kpts, None)

		# compute matches
		pts1, pts2 = [], []
		if self.sourse_des is not None:

			matches = self.mtch.knnMatch(self.sourse_des, des, k=2)
			# matches = sorted(matches, key=lambda x: x.distance)
			for m, n in matches:
				if m.distance < 0.75 * n.distance:
					# good.append(m)
					pts2.append(kpts[m.trainIdx].pt)
					pts1.append(self.sourse_kpts[m.queryIdx].pt)
			F, mask = cv2.findFundamentalMat(np.asarray(pts1), np.asarray(pts2), cv2.FM_8POINT)
			print(F)
		self.sourse_des = des
		self.sourse_kpts = kpts
		return pts2, pts1

class Fundamental():
	def __init__(self):
		pass
	def get_fundamental(self, pts1, pts2):
		F = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)
		return F