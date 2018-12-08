import sys

sys.path.append("/home/sviatoslavpovod/Documents/tools/pangolin")
import OpenGL.GL as gl
import pangolin

import numpy as np


def main(pts=[], dcam=0, scam=0):
	# pangolin.CreateWindowAndBind('Main', 640, 480)
	# gl.glEnable(gl.GL_DEPTH_TEST)
	#
	# # Define Projection and initial ModelView matrix
	# scam = pangolin.OpenGlRenderState(
	# 	pangolin.ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 200),
	# 	pangolin.ModelViewLookAt(-2, 2, -2, 0, 0, 0, pangolin.AxisDirection.AxisY))
	# handler = pangolin.Handler3D(scam)
	#
	# # Create Interactive View in window
	# dcam = pangolin.CreateDisplay()
	# dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -640.0 / 480.0)
	# dcam.SetHandler(handler)




	if True:
		gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
		gl.glClearColor(1.0, 1.0, 1.0, 1.0)
		dcam.Activate(scam)

		# Render OpenGL Cube
		pangolin.glDrawColouredCube(0.1)

		# Draw Point Cloud
		# points = np.random.random((10000, 3)) * 3 - 4
		# gl.glPointSize(1)
		# gl.glColor3f(1.0, 0.0, 0.0)
		# pangolin.DrawPoints(points)
		if len(pts) > 0:
			gl.glPointSize(2)
			pangolin.DrawPoints(pts)

		# Draw Point Cloud
		# points = np.random.random((10000, 3))
		# colors = np.zeros((len(points), 3))
		# colors[:, 1] = 1 - points[:, 0]
		# colors[:, 2] = 1 - points[:, 1]
		# colors[:, 0] = 1 - points[:, 2]
		# points = points * 3 + 1
		# gl.glPointSize(1)
		# pangolin.DrawPoints(points, colors)

		pangolin.FinishFrame()


if __name__ == '__main__':
	main()