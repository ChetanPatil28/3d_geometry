import cv2
import numpy as np
import utils
from utils import Camera

np.random.seed(79)
np.set_printoptions(precision=3)


pts3d_11 = np.random.randint(11, 50, size=(8, 3)).astype(np.float32) # 8-points
K = np.load("../camMatrix_720p.npy")
dist = np.zeros(shape=5)

# In this file, we shall see how can we estimate the Relative-SCale.
# I see this estimation as a Catch-22 situation.
# But not sure if it really is.

# Lets see whats the situation looks like

# Technically, in Vis-Odom, the P we get from Essential mat has t of unit-lenght (it only has direction, no mag).
# So in order to estimate the actual maginitude, we actually use the ratio formula.
# Refer this
# 1. https://robotics.stackexchange.com/questions/14471/relative-scale-in-sfm
# 2. https://docs.google.com/viewer?url=http%3A%2F%2Frpg.ifi.uzh.ch%2Fdocs%2FVO_Part_I_Scaramuzza.pdf
# 3. https://stackoverflow.com/questions/58259795/how-to-calculate-the-relative-scale-from-two-fundamental-matrices-that-share-a-c

# See that this formula involves triangulation, but to get triangulation, we need P, but P's t does not have
# any magnitude only.
# See the problem here ?

# Actually, it turns out that, whatever the t's scale is, a 3D point will still end up at the same 2D location only.

# TO BE PROVED
# Take three cameras with any arbitrary place.
# Take three more camers with same config except that their t is unit-length.

# Verify if u get the distance correctly in the 2nd camera systems.

# These will be the centres.
c1 = np.asarray([0, 0, 0]).reshape(3, -1)
c2 = np.asarray([70, 70, 70]).reshape(3, -1)
c3 = np.asarray([30, 15, 40]).reshape(3, -1)
c4 = np.asarray([25, 45, 25]).reshape(3, -1)

# Let these be the first set of cameras
Cam1 = Camera(K, Rc=utils.rotate(), center=c1)
pts2d_1 = utils.hom_to_euc(utils.project(Cam1.P, pts3d_11))
Cam2 = Camera(K, Rc=utils.rotate(thetay=169), center=c2)
pts2d_2 = utils.hom_to_euc(utils.project(Cam2.P, pts3d_11))
Cam3 = Camera(K, Rc=utils.rotate(thetaz=90), center=c3)
pts2d_3 = utils.hom_to_euc(utils.project(Cam3.P, pts3d_11))
Cam4 = Camera(K, Rc=utils.rotate(thetax=90), center=c4)
pts2d_4 = utils.hom_to_euc(utils.project(Cam4.P, pts3d_11))


# Let these be the second set of cameras at just unit-distance from world
UnitCam1 = Camera(K, Rc=utils.rotate(), center=utils.norm(c1))
unit_pts2d_1 = utils.hom_to_euc(utils.project(UnitCam1.P, pts3d_11))
UnitCam2 = Camera(K, Rc=utils.rotate(thetay=169), center=utils.norm(c2))
unit_pts2d_2 = utils.hom_to_euc(utils.project(UnitCam2.P, pts3d_11))
UnitCam3 = Camera(K, Rc=utils.rotate(thetaz=90), center=utils.norm(c3))
unit_pts2d_3 = utils.hom_to_euc(utils.project(UnitCam3.P, pts3d_11))
UnitCam4 = Camera(K, Rc=utils.rotate(thetax=90), center=utils.norm(c4))
unit_pts2d_4 = utils.hom_to_euc(utils.project(UnitCam4.P, pts3d_11))

# TODO
# lets create a class which will estimate the actual t12 for us.
# It should need three consecutive set of 2D points
# For example, if we give pts2d_1, pts2d_2, pts2d_3, it'll estimate the magnitude of `t12` for us.
# Do this next week

# Before doing this, research about Trifocal Tensor once in HZ. MVG

######### short experiment

# let's triangulate the frst two normal cams
# then triangulate the second set of unit-cameras.


print("~~~~~~ ", (Cam2.P - UnitCam2.P).sum(), (Cam3.P - UnitCam3.P).sum())

pts4d = cv2.triangulatePoints(Cam2.P, Cam3.P, pts2d_2.T, pts2d_3.T).T
pts4d = (pts4d/pts4d[:, -1].reshape(-1, 1))[:, :-1]


# print("pts normal ", pts4d)

unit_pts4d = cv2.triangulatePoints(UnitCam2.P, UnitCam3.P, unit_pts2d_2.T, unit_pts2d_3.T).T
unit_pts4d = (unit_pts4d/unit_pts4d[:, -1].reshape(-1, 1))[:, :-1]

# print("pts normal ", unit_pts4d)


print("@@@@ ", Cam2.Rt,"\n\n", UnitCam2.Rt, np.linalg.norm(UnitCam2.t))


#########

# R12s, t12s = cv2.findEssentialMat(pts2d_1, pts2d_2, K, cv2.FM_RANSAC)

#### VERY IMPORTANT.
# actually in reality, the projection matrix's t  (got from essential-mat) entry is of unit-length,
# lets use the normed one

# cam11_p = Cam11.P.copy()
# cam11_p[:3, 3] = utils.norm(Cam11.t).flatten()
#
# cam12_p = Cam12.P.copy()
# cam12_p[:3, 3] = utils.norm(Cam12.t).flatten()
#
# cam13_p = Cam13.P.copy()
# cam13_p[:3, 3] = utils.norm(Cam13.t).flatten()
#
# pts4d = cv2.triangulatePoints(cam11_p, cam11_p, pts2d_11.T, pts2d_11.T).T
# print("#### ", pts4d)
# print("Triangulation 1 with norm-t ", ((pts4d/pts4d[:, -1].reshape(-1, 1))[:, :-1] - pts3d_11).sum())
# pts4d = cv2.triangulatePoints(cam12_p, cam13_p, pts2d_12.T, pts2d_13.T).T
# print("Triangulation 2 with norm-t ", ((pts4d/pts4d[:, -1].reshape(-1, 1))[:, :-1] - pts3d_11).sum())


# print("####### ", Cam2.t/np.linalg.norm(Cam2.t), np.linalg.norm(Cam2.t), UnitCam2.t * np.linalg.norm(Cam2.t) )
#
#
# print("**** ", unit_pts2d_2)
#
# print("---- ", pts2d_2)