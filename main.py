import numpy as np
import cv2
import os
from FeatureExtractor import FeatureExtractor
dir = "/home/sviatoslavpovod/Downloads/mav0/cam0/data/"
cap = cv2.VideoCapture(0)

im_list = sorted(os.listdir(dir))

feate = FeatureExtractor()

prevIm, prevKpt = 0, 0
i = 0
while True:

	for imname in im_list:
		im = cv2.imread(dir + imname)
		pts2, pts1 = feate.extract(im)
		if len(pts1) > 0:
			for p1, p2 in zip(pts1, pts2):
				im = cv2.line(im, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color=(0,255,0))
		cv2.imshow("sdc", im)
		cv2.waitKey(1)