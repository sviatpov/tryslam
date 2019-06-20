import sys
sys.path.append("/home/sviat/Documents/lib/g2opy/env/lib/python3.6/site-packages/")
import g2o
import numpy as np




class Point():

    def __init__(self, coord):
        self.pt = coord
        self.frames = []
        self.idx = []
        self.id = None

    ## TODO : how to change it on numpy array

    def add_observation(self, frame, idx):
        frame.pts[idx] = self
        self.frames.append(frame)
        self.idx.append(idx)


    # def add_observation(self, frame, idx):



class Frame():
    def __init__(self, im):
        self.im = im
        self.key_pts = None
        self.des = None
        self.idx = None
        self.pts = None
        self.pose = None

    def add_feature(self, f):
        self.key_pts = f[0]
        self.des = f[1]
        self.pts = np.zeros(shape=(len(self.key_pts),), dtype=np.bool)

class Map():
    def __init__(self):
        self.frames = []
        self.points = []
        self.state = None
        self.q = None
        self.D4 = np.array([0, 0, 0, 0]).reshape(4, 1)


    ## Optimization part
    def append_frame(self, frame):
        frame.idx = (len(self.frames) + 1)
        self.frames.append(frame)

    def get_back_frame(self):
        return self.frames[-1]

    def get_two_back_frame(self):
        return self.frames[-2], self.frames[-1]

    def append_point(self, point):
        point.id = len(self.points) + 1
        self.points.append(point)

    def add_valid_points(self, idx1, idx2, D4):
        self.frames[-2].pts[idx1] = True
        self.frames[-1].pts[idx2] = True

    def oprimizer(self):

        ## Init optimizer
        optimizer = g2o.SparseOptimizer()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        optimizer.set_algorithm(solver)


        ## add frames to graph
        for f in self.frames:
            v_se3 = g2o.VertexSE3()
            v_se3.set_id(f.id)
            v_se3.set_estimate(f.pose)
            v_se3.set_fixed(f.id == 0)
            optimizer.add_vertex(v_se3)

