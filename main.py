import numpy as np
import sys
sys.path.append("/home/sviatoslavpovod/Documents/tools/pangolin")
import OpenGL.GL as gl
import pangolin

import cv2
import os
from FeatureExtractor import FeatureExtractor
from undistort import undistort, K
from Pangolin import main
dir = "/home/sviatoslavpovod/Downloads/mav0/cam0/data/"

pangolin.CreateWindowAndBind('Main', 640, 480)
gl.glEnable(gl.GL_DEPTH_TEST)

# Define Projection and initial ModelView matrix
scam = pangolin.OpenGlRenderState(
pangolin.ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 200),
pangolin.ModelViewLookAt(-2, 2, -2, 0, 0, 0, pangolin.AxisDirection.AxisY))
handler = pangolin.Handler3D(scam)

# Create Interactive View in window
dcam = pangolin.CreateDisplay()
dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -640.0 / 480.0)
dcam.SetHandler(handler)

# cap = cv2.VideoCapture(4)
im_list = sorted(os.listdir(dir))
feate = FeatureExtractor(K)
f1 = 0

while True:
	for imname in im_list:
		im = cv2.imread(dir + imname)
		im = undistort(im)
		# _, im = cap.read(cv2.IMREAD_GRAYSCALE)
		f2 = feate.extract(im)
		if f1 != 0:
			pts2, pts1 = feate.matches_frame(f1, f2)
			for p1, p2 in zip(pts1, pts2):
				im = cv2.line(im, (int(p1[0]), int(p1[1])), \
							  (int(p2[0]), int(p2[1])), color=(0,255,0))
			R, T = feate.getRT(pts1, pts2)
			D4 = feate.get3D(R, T, pts1.T, pts2.T).T
			D4 = D4[:, 0:3] / D4[:, 3:4]
			mask = D4[:, 2] > 0.05
			D4 = D4[mask]
			main(D4,dcam, scam)
		f1 = f2
		cv2.imshow("sdc", im)
		# cv2.imshow("sdc2", imd)
		cv2.waitKey(1)