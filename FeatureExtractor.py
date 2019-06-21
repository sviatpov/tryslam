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
		self.F = np.eye(3)
		self.K = K

	def getRT(self, pts1, pts2):

		## TODO camera matrix remove from class field
		E = self.K.T @ self.F @ self.K
		_, R, T, mask = cv2.recoverPose(E, pts1, pts2, self.K)

		return R, T, np.asarray(mask[:,0], dtype=np.bool)

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
				if m.distance < 30:
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


	def filter_by_mask(self, *args, mask):

		ret = []
		for e in args:
			ret.append(e[mask])
		return ret

	def estimate_pose(self, f1, f2, save_pose = True):

		## match features on two images

		idx1, pts1, idx2, pts2 = self.matches_frame(f1, f2)

		## estimate transformation from matched pixels
		R, T, mask = self.getRT(pts1, pts2)
		idx1, idx2, pts1, pts2 = self.filter_by_mask(idx1, idx2, pts1, pts2, mask=mask)
		## save pose into

		if save_pose:
			f2.pose = f1.pose @ Rt4_inv(R, T)

############# TODO optimize this part
		idx_mask_where_true = np.zeros(shape=len(idx1), dtype=np.bool)

		idx_mask_where_true[np.where(f1.pts_mask[idx1] == True)] = True



		idx2_mask = idx2[idx_mask_where_true]
		idx1_mask = idx1[idx_mask_where_true]
		for i, j in zip(idx2_mask, idx1_mask):
			f1.pts[j].add_observation(f2, i)
####################################################################################


		D4 = self.get3D(R, T, pts1.T, pts2.T)

		## TODO check do we need it
		D4 = D4.T
		D4 = D4 / D4[:,-1, np.newaxis]
		D4, idx1, idx2, pts2 = self.filter_by_mask(D4, idx1, idx2,pts2, mask=D4[:, -2] > 0.05)

		f1.pts_mask[idx1] = True
		f2.pts_mask[idx2] = True
		return idx1, idx2, D4, pts2

def poseRt(R, t):
  ret = np.eye(4)
  ret[:3, :3] = R
  ret[:3, 3] = t
  return ret

def optimize(frames, points, local_window, fix_points, verbose=False, rounds=50):
	if local_window is None:
		local_frames = frames
	else:
		local_frames = frames[-local_window:]

	# create g2o optimizer
	opt = g2o.SparseOptimizer()
	solver = g2o.BlockSolverSE3(g2o.LinearSolverCSparseSE3())
	solver = g2o.OptimizationAlgorithmLevenberg(solver)
	opt.set_algorithm(solver)

	# add normalized camera
	cam = g2o.CameraParameters(1.0, (0.0, 0.0), 0)
	cam.set_id(0)
	opt.add_parameter(cam)

	robust_kernel = g2o.RobustKernelHuber(np.sqrt(5.991))
	graph_frames, graph_points = {}, {}

	# add frames to graph
	for f in (local_frames if fix_points else frames):
		pose = f.pose
		se3 = g2o.SE3Quat(pose[0:3, 0:3], pose[0:3, 3])
		v_se3 = g2o.VertexSE3Expmap()
		v_se3.set_estimate(se3)

		v_se3.set_id(f.id * 2)
		v_se3.set_fixed(f.id <= 1 or f not in local_frames)
		# v_se3.set_fixed(f.id != 0)
		opt.add_vertex(v_se3)

		# confirm pose correctness
		est = v_se3.estimate()
		assert np.allclose(pose[0:3, 0:3], est.rotation().matrix())
		assert np.allclose(pose[0:3, 3], est.translation())

		graph_frames[f] = v_se3

	# add points to frames
	for p in points:
		if not any([f in local_frames for f in p.frames]):
			continue

		pt = g2o.VertexSBAPointXYZ()
		pt.set_id(p.id * 2 + 1)
		pt.set_estimate(p.pt[0:3])
		pt.set_marginalized(True)
		pt.set_fixed(fix_points)
		opt.add_vertex(pt)
		graph_points[p] = pt

		# add edges
		for f, idx in zip(p.frames, p.idxs):
			if f not in graph_frames:
				continue
			edge = g2o.EdgeProjectXYZ2UV()
			edge.set_parameter_id(0, 0)
			edge.set_vertex(0, pt)
			edge.set_vertex(1, graph_frames[f])
			edge.set_measurement(f.kps[idx])
			edge.set_information(np.eye(2))
			edge.set_robust_kernel(robust_kernel)
			opt.add_edge(edge)

	if verbose:
		opt.set_verbose(True)
	opt.initialize_optimization()
	opt.optimize(rounds)

	# put frames back
	for f in graph_frames:
		est = graph_frames[f].estimate()
		R = est.rotation().matrix()
		t = est.translation()
		f.pose = poseRt(R, t)

	# put points back
	if not fix_points:
		for p in graph_points:
			p.pt = np.array(graph_points[p].estimate())

	return opt.active_chi2()
