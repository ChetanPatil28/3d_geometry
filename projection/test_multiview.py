import cv2
import numpy as np
import utils
np.random.seed(79)
# np.set_printoptions(precision=4)
np.set_printoptions(precision=3)

Points = np.random.randint(11, 50, size=(8, 3)) # 8-points
K = np.load("../camMatrix_720p.npy")
dist = np.zeros(shape=5)


class Camera:
    def __init__(self,K=np.eye(3), R=np.eye(3), center=np.zeros((3,1))):
        self.K = K
        self.R = R
        self.center = center
        self.t = -self.R.dot(self.center)
        self.Rt = np.hstack((self.R, self.t))
        self.P = utils.P_from_krt(self.K, self.R, self.t)

    def norm(self, vec):
        return vec/np.linalg/norm(vec)


########################################## Case-1 ########################################## 
## Here we shall place two cameras, one at origin and the other somehwere ie R2 and X_cam2.
## We shall manually project a set of 3d points on two different cameras.
## We shall use the 2d-projected points from both the cameras, calculate the Essential-Mat, get the R and t.
## This R and t from essential must be same as our predefined R2, t2.

# This is the first camera with same orientation and same place as origin.
Cam11 = Camera(K)
pts2d_11 = utils.project(Cam11.P, Points)
pts2d_11 = utils.hom_to_euc(pts2d_11)

# This is second camera.
Cam12 = Camera(K, R=utils.rotate(thetax=49), center=np.asarray([10, 25, 7]).reshape(3, -1))
Cam12_R = Cam12.R
Cam12_c = Cam12.center
Cam12_t = Cam12.t
pts2d_12 = utils.project(Cam12.P, Points)
pts2d_12 = utils.hom_to_euc(pts2d_12)

# This is third camera.
Cam13 = Camera(K, R=utils.rotate(thetay=68), center=np.asarray([15, 25, 10]).reshape(3, -1))
Cam13_R = Cam13.R
Cam13_c = Cam13.center
Cam13_t = Cam13.t
pts2d_13 = utils.project(Cam13.P, Points)
pts2d_13 = utils.hom_to_euc(pts2d_13)

Ess12, _ = cv2.findEssentialMat(pts2d_11, pts2d_12, Cam12.K, cv2.FM_RANSAC)
R12_est1, R12_est2, t12_est = cv2.decomposeEssentialMat(Ess12) # t2_est shall be a unit-vector. it wont give the legth, but only dir.
print("R estimated is equal ???  ", (Cam12_R - R12_est1).sum(), np.allclose(Cam12_R, R12_est2))
print("Is the t equal ??? ", np.allclose(utils.norm(Cam12_t), t12_est))



########################################## Case-2 ########################################## 
## Here we shall place two cameras, one with R2, t2 and other with R3, t3
## We shall manually project a set of 3d points on these two cameras.
## we shall do the same things as we did above.
## Except that these R  and t will be the ones wrt the first-camera.


Ess23, _ = cv2.findEssentialMat(pts2d_12, pts2d_13, K ,cv2.FM_RANSAC)
R23_est1, R23_est2, t23_est = cv2.decomposeEssentialMat(Ess23) # t23_est shall be a unit-vector. it wont give the legth, but only direction.
## Note that these R23_est1 or R23_est2 will give you the rotation of Cam3 wrt to Cam2. (assuming that cam2 was at rotation=0).


# #### Now let's see how we can get the rotation of Cam3 wrt Cam1(the original origin).
# ## One way to get this is by findng the essential matrix and then decomposing using pts2d_1 and pts2d_3.
# # Any other way ?  HOW TO DO THIS ?
# # wkt, R2 orients from cam1(origin) to cam2. And R3_est1 orients from cam2 to cam3.
# # Then,we can simply orient from cam3 to cam2 using `R3_est1.transpose()`. and then from cam2 to cam1 using (R2_est1.transpose())


# # But how do we know if the above steps are correct. Just use essential mat from `pts2d_1` and `pts2d_3` ,
# # get the rotation-mat and verify.
#
# Ess3, _ = cv2.findEssentialMat(pts2d_1, pts2d_3, K ,cv2.FM_RANSAC)
# R13_est1, R13_est2, t13_est = cv2.decomposeEssentialMat(Ess3) # t_est shall be a unit-vector. it wont give the legth, but only direction.
#
# # lets see in code.
# # R_13 means orientation of cam3 wrt cam1. Think why there is .T in the end :-)
# R_13 = np.matmul(R2_est1.T, R3_est1.T).T
# print((R13_est1 - R_13).sum())
# # As we see, the difference between both is of order 10e-8.
#
#
#
# ## Brainstorm question.
# # R1 orients from Cam1 to Cam2,
# # R2 orients from Cam2 to Cam3.
# # R3 orients from Cam1 to Cam3.
#
# # U are given only R1 and R3 , find R2.
#
# # print("R2 is ", np.matmul(R3_est1, R13_est1.T), "\n", R2)
#
#
#
#
#
# #### Now, we are done with ROTATIONS. Lets now deal with TRANSLATIONS.
#
#
# ############################################
#
# # Lets get cam-centre directions of all cameras from the estimated ts.
#
#
# Cam2_est = -R2_est1.T.dot(t2_est)
# Cam3_est = -R13_est1.T.dot(t13_est)
# print("Cam2 \n", Cam2_est, np.allclose(Cam2_est, X_cam2/np.linalg.norm(X_cam2)))
# print("Cam3 \n", Cam3_est, np.allclose(Cam3_est, X_cam3/np.linalg.norm(X_cam3)))
#
# # print("t3-est ", t3_est)
#
#
#
#
# # How do we get Cam3-direction wrt Cam2 ?????
# # Simple vector addition. ie cam2  + cam23  = cam3.
# Cam23_est = -R2.T.dot(-R3_est1.T.dot(t3_est))
# X_cam23 = X_cam3 - X_cam2
# print("Xcam23-norm will be ", (X_cam23/np.linalg.norm(X_cam23) - Cam23_est).sum())
#
# # print("Cam3 wrt cam2 ",Cam23_est, R3_est1.dot(Cam2_est), Cam3_est - Cam2_est )
#
#
#
# # Let's see what will be the dir of Cam3 wrt Cam1.
# # It should actually be
#
#
# # Previously we saw that, t3_est was dir-vector from Cam2 pointing towards Cam3.
# # Can we get the dir-vec of Cam2 wrt to Cam1.???
# # We actually know this dir-vec. This will be norm(X_cam2).
# # But lets try to get it the other-way around.
#
# # We can get this by -R2_est.transpose * t3_est.
