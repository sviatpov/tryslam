

import sys
sys.path.append("/home/sviat/Documents/lib/g2opy/env/lib/python3.6/site-packages/")
import g2o
import numpy as np



class Point():
    def __init__(self, map, loc, col=None, tid=None):
        self.pt = loc
        # frames on which we can see this point
        self.frames = []
        self.idx = []
        self.color = col
        # self.id = tid if tid is not None


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
        self.pts = np.zeros(shape=(len(self.key_pts), 3), dtype=np.bool)

class Map():
    def __init__(self):
        self.frames = []
        self.points = []
        self.state = None
        self.q = None


    ## Optimization part
    def append_frame(self, frame):
        frame.idx = (len(self.frames) + 1)
        self.frames.append(frame)

    def get_back_frame(self):
        return self.frames[-1]

    def get_two_back_frame(self):
        return self.frames[-2], self.frames[-1]

    def append_points(self, points):
        self.points.append(points)

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

