import sys
# path to compiled pangolin.so
sys.path.append("/home/sviatoslavpovod/Documents/tools/pangolin")
import OpenGL.GL as gl
import pangolin


class vizualizer():
	def __init__(self):
		pangolin.CreateWindowAndBind('Main', 640, 480)
		gl.glEnable(gl.GL_DEPTH_TEST)

		# Define Projection and initial ModelView matrix
		self.scam = pangolin.OpenGlRenderState(
		pangolin.ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 200),
		pangolin.ModelViewLookAt(-2, 2, -2, 0, 0, 0, pangolin.AxisY))
		self.tree = pangolin.Renderable()
		self.tree.Add(pangolin.Axis())
		handler = pangolin.Handler3D(self.scam)

		# Create Interactive View in window
		self.dcam = pangolin.CreateDisplay()
		self.dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -640.0 / 480.0)
		self.dcam.SetHandler(handler)
		self.dcam.SetDrawFunction(self._draw)

	def _draw(self,view):
		view.Activate(self.scam)
		self.tree.Render()

	def main(self, pts=[], pts2=[]):

		if True:
			gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
			gl.glClearColor(1.0, 1.0, 1.0, 1.0)
			self.dcam.Activate(self.scam)

			# Render OpenGL Cube
			# pangolin.glDrawColouredCube(0.1)

			# Draw Point Cloud
			gl.glColor3f(1.0, 0.0, 0.0)
			# pangolin.DrawPoints(points)
			if len(pts) > 0:
				gl.glPointSize(4)
				pangolin.DrawPoints(pts)
				gl.glColor3f(0.0, 0.0, 0.0)
				gl.glPointSize(1)
				pangolin.DrawPoints(pts2)

			pangolin.FinishFrame()


if __name__ == '__main__':
	vizualizer()