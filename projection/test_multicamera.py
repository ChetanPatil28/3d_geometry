import cv2
import numpy as np
import utils
from utils import Camera

np.random.seed(79)
np.set_printoptions(precision=3)

pts3d_11 = np.random.randint(11, 50, size=(8, 3))  # 8-points
K = np.load("../camMatrix_720p.npy")
dist = np.zeros(shape=5)

# This is the first camera with same orientation and same place as origin.
Cam11 = Camera(K)
pts2d_11 = utils.project(Cam11.P, pts3d_11)
pts2d_11 = utils.hom_to_euc(pts2d_11)

# This is second camera.
Cam12 = Camera(K, Rc=utils.rotate(thetax=49), center=np.asarray([10, 25, 7]).reshape(3, -1))
Cam12_R = Cam12.R
Cam12_c = Cam12.center
Cam12_t = Cam12.t
pts2d_12 = utils.project(Cam12.P, pts3d_11)
pts2d_12 = utils.hom_to_euc(pts2d_12)

# This is third camera.
Cam13 = Camera(K, Rc=utils.rotate(thetay=68), center=np.asarray([15, 25, 10]).reshape(3, -1))
Cam13_R = Cam13.R
Cam13_c = Cam13.center
Cam13_t = Cam13.t
pts2d_13 = utils.project(Cam13.P, pts3d_11)
pts2d_13 = utils.hom_to_euc(pts2d_13)

Ess12, _ = cv2.findEssentialMat(pts2d_11, pts2d_12, Cam12.K, cv2.FM_RANSAC)
R12_est1, R12_est2, t12_est = cv2.decomposeEssentialMat(Ess12)

Ess13, _ = cv2.findEssentialMat(pts2d_11, pts2d_13, K, cv2.FM_RANSAC)
R13_est1, R13_est2, t13_est = cv2.decomposeEssentialMat(Ess13)

Ess23, _ = cv2.findEssentialMat(pts2d_12, pts2d_13, K, cv2.FM_RANSAC)
R23_est1, R23_est2, t23_est = cv2.decomposeEssentialMat(Ess23)

print("R estimated is equal ???  ", (Cam12_R - R12_est1).sum(), (Cam12_R - R12_est2).sum())
print("Is the t equal ??? ", (utils.norm(Cam12_t) - t12_est).sum())

# Q1. What is the location of the `Points` as seen by the Cam3 ??
# Ans. P' = [R | t] * P
pts3d_13 = np.matmul(Cam13.Rt, utils.euc_to_hom(pts3d_11).T)  # [3, N]
# How to verify if the above formula is correct ?.
# Well, reverse-transform to get the original points ie P = [R.T | -R.T*t] * P'
pts3d_11_est = np.matmul(np.hstack((Cam13.R.T, -Cam13.R.T.dot(Cam13.t))), utils.euc_to_hom(pts3d_13.T).T).T
# Let's see if the estimated and the original are same.
print("Are both same ", (pts3d_11 - pts3d_11_est).sum())
