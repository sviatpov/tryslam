import sys
import numpy as np
from multiprocessing import Process, Queue, set_start_method
# path to compiled pangolin.so
sys.path.append("/home/sviat/Documents/apps/pangolin_for_python/pangolin")
import OpenGL.GL as gl
import pangolin


class vizualizer():
	def __init__(self):
		pangolin.CreateWindowAndBind('Main', 640, 480)
		gl.glEnable(gl.GL_DEPTH_TEST)

		# Define Projection and initial ModelView matrix
		self.scam = pangolin.OpenGlRenderState(
		pangolin.ProjectionMatrix(640, 480, 520, 520, 320, 240, 0.2, 200),
		pangolin.ModelViewLookAt(0, 1, -3, 0, 0, 0, pangolin.AxisY))
		self.tree = pangolin.Renderable()
		self.tree.Add(pangolin.Axis())
		handler = pangolin.Handler3D(self.scam)

		# Create Interactive View in window
		self.dcam = pangolin.CreateDisplay()
		self.dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -640.0 / 480.0)
		self.dcam.SetHandler(handler)
		self.dcam.SetDrawFunction(self._draw)

		self.pts_prev = [0,0,0]
	def _draw(self,view):
		view.Activate(self.scam)
		self.tree.Render()

	def draw_cameras(self, cam):
		gl.glColor3f(0.0, 1.0, 0.0)
		for c in cam:
			pangolin.DrawCamera(c, 0.35, 0.75, 0.8)

	def main(self, pts=[], colors=None, camera=None):

		if True:
			gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
			gl.glClearColor(1.0, 0.5, 0.5, 1.0)
			self.dcam.Activate(self.scam)

			# Render OpenGL Cube
			# pangolin.glDrawColouredCube(0.1)

			# Draw Point Cloud
			gl.glColor3f(1.0, 1.0, 0.0)
			# pangolin.DrawPoints(points)
			if len(pts) > 0:
				gl.glPointSize(3)
				if isinstance(colors, np.ndarray):
					pangolin.DrawPoints(pts, colors)
				else:
					pangolin.DrawPoints(pts)

				if isinstance(camera, list):
					self.draw_cameras(camera)
				# gl.glColor3f(0.0, 0.0, 0.0)
				# gl.glPointSize(1)
				# pangolin.DrawPoints(pts2)

			pangolin.FinishFrame()

def draw_process(q):
	v = vizualizer()
	pts_prev = [[0,0,0]]
	pts = {}

	while True:
		# print('dfvdf')
		try:
			pts = q.get(block=False)
		except:
			pass

		cloud = pts.get('points')
		colors = pts.get('colors')
		cam = pts.get('cameras')
		if isinstance(cloud, np.ndarray):
			pts_prev = cloud
		v.main(pts_prev, colors, cam)


# def f(q):
# 	for i in range(100):
# 		q.put(i)
#
# if __name__ == '__main__':
#
#
#
#
#
# 	set_start_method('spawn')
# 	q = Queue()
# 	p = Process(target=f, args=(q,))
# 	p.start()
#
# 	while True:
# 		print(q.get())
# 	p.join()