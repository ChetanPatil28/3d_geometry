import cv2
import numpy as np
import utils
from utils import Camera

# np.random.seed(79)
np.set_printoptions(precision=3)

thetaX = np.random.randint(0, 360)
thetaY = np.random.randint(0, 360)
thetaZ = np.random.randint(0, 360)

pts3d_11 = np.random.randint(11, 50, size=(8, 3)).astype(np.float32) # 8-points
K = np.load("../camMatrix_720p.npy")
dist = np.zeros(shape=5)

# This is the first camera with same orientation and same place as origin.
Cam11 = Camera(K)
pts2d_11 = utils.project(Cam11.P, pts3d_11)
pts2d_11 = utils.hom_to_euc(pts2d_11)

# This is second camera.
Cam12 = Camera(K, Rc=utils.rotate(thetax=thetaX), center=np.asarray([10, 25, 7]).reshape(3, -1))
Cam12_R = Cam12.R
Cam12_c = Cam12.center
Cam12_t = Cam12.t
pts3d_12 = np.matmul(Cam12.Rt, utils.euc_to_hom(pts3d_11).T).T
pts2d_12 = utils.project(Cam12.P, pts3d_11)
pts2d_12 = utils.hom_to_euc(pts2d_12)

# This is third camera.
Cam13 = Camera(K, Rc=utils.rotate(thetay=thetaY), center=np.asarray([15, 25, 10]).reshape(3, -1))
Cam13_R = Cam13.R
Cam13_c = Cam13.center
Cam13_t = Cam13.t
pts3d_13 = np.matmul(Cam13.Rt, utils.euc_to_hom(pts3d_11).T).T
pts2d_13 = utils.project(Cam13.P, pts3d_11)
pts2d_13 = utils.hom_to_euc(pts2d_13)

# From here on, we shall test the findEssentialMat method.

Ess12, _ = cv2.findEssentialMat(pts2d_11, pts2d_12, Cam12.K, cv2.FM_RANSAC)
R12_est1, R12_est2, t12_est = cv2.decomposeEssentialMat(Ess12) # It gives orientation of Cam2 wrt Cam1.

Ess13, _ = cv2.findEssentialMat(pts2d_11, pts2d_13, K, cv2.FM_RANSAC)
R13_est1, R13_est2, t13_est = cv2.decomposeEssentialMat(Ess13) # It gives orientation of Cam3 wrt Cam1.

Ess23, _ = cv2.findEssentialMat(pts2d_12, pts2d_13, K, cv2.FM_RANSAC)
R23_est1, R23_est2, t23_est = cv2.decomposeEssentialMat(Ess23) # It gives orientation of Cam3 wrt Cam2.

print("R estimated is equal ???  ", (Cam12_R - R12_est1).sum(), (Cam12_R - R12_est2).sum())
print("Is the t equal ??? ", (utils.norm(Cam12_t) - t12_est).sum())

# Q1. What is the location of the `Points` as seen by the Cam3 ??
# Ans. P' = [R | t] * P
pts3d_13_est = np.matmul(Cam13.Rt, utils.euc_to_hom(pts3d_11).T)  # [3, N]
# How to verify if the above formula is correct ?.
# Well, reverse-transform to get the original points ie P = [R.T | -R.T*t] * P'
pts3d_11_est = np.matmul(np.hstack((Cam13.R.T, -Cam13.R.T.dot(Cam13.t))), utils.euc_to_hom(pts3d_13_est.T).T).T
# Let's see if the estimated and the original are same.
print("Switching between different frames ", (pts3d_11 - pts3d_11_est).sum())

# Q2. Given Cam1, 2 and 3 , can you calculate R13 ie orientation of Cam1 wrt Cam3 ?
# Ans. You need to move from Cam1 to Cam2 and then from Cam2 to Cam3. This gives us `Cam1 wrt Cam3` ie R13.
print("Estimating Camera orientations 1. -- ",
      (np.matmul(R23_est1, Cam12.R) - Cam13.R).sum(), (np.matmul(R23_est2, Cam12.R) - Cam13.R).sum())

# Q3. Can you verify if the estimated R23 is correct ?
# Ans. Move from Cam2 to Cam1 , and them from Cam1 to Cam3. This gives us `Cam2 wrt Cam3` ie R23.
print("Estimating Camera orientations 2. -- ", (np.matmul(Cam13.R, Cam12.R.T) - R23_est1).sum(),
                                               (np.matmul(Cam13.R, Cam12.R.T) - R23_est2).sum())


# From here on, we shall test the solvePnP method.
# Example-1.
val, rvec, t11_pnp_est, inliers = cv2.solvePnPRansac(pts3d_11, pts2d_11, K, None, None, None,
                                                False, 50, 2.0, 0.99, None)
R11_pnp_est, _ = cv2.Rodrigues(rvec)

# Example-2.
val, rvec, t12_pnp_est, inliers = cv2.solvePnPRansac(pts3d_11, pts2d_12, K, None, None, None,
                                                False, 50, 2.0, 0.99, None)
R12_pnp_est, _ = cv2.Rodrigues(rvec)

# Example-3.
val, rvec, t13_pnp_est, inliers = cv2.solvePnPRansac(pts3d_11, pts2d_13, K, None, None, None,
                                                False, 50, 2.0, 0.99, None)
R13_pnp_est, _ = cv2.Rodrigues(rvec)

print("Solve-PnP Results ", (Cam12.R - R12_pnp_est).sum(),  (Cam12.t - t12_pnp_est).sum())

# Example-4.
val, rvec, t23_pnp_est, inliers = cv2.solvePnPRansac(pts3d_12, pts2d_13, K, None, None, None,
                                                False, 50, 2.0, 0.99, None)
R23_pnp_est, _ = cv2.Rodrigues(rvec)

print("R23 - - - - ", R23_pnp_est, R23_est1)
print("t23 - - - - ", utils.norm(t23_pnp_est), t23_est)