import cv2
import numpy as np
import utils
np.random.seed(79)
# np.set_printoptions(precision=4)
np.set_printoptions(precision=3)

Points = np.random.randint(11, 50, size=(8, 3)) # 8-points
K = np.load("../camMatrix_720p.npy")
dist = np.zeros(shape=5)



########################################## Case-1 ########################################## 
## Here we shall place two cameras, one at origin and the other somehwere ie R2 and X_cam2.
## We shall manually project a set of 3d points on two different cameras.
## We shall use the 2d-projected points from both the cameras, calculate the Essential-Mat, get the R and t.
## This R and t from essential must be same as our predefined R2, t2.

# This is the first camera with same orientation and same place as origin.
X_cam1 = np.asarray([0, 0, 0]).reshape(3, -1)
R1 = utils.rotate()
t1 = -R1.dot(X_cam1) # t will be R_transpose.cam-centre

P1 = utils.P_from_krt(K, R1, t1)
pts2d_1 = utils.project(P1, Points)
pts2d_1 = utils.hom_to_euc(pts2d_1)
# print("Finally points are ", pts2d_1)
    

# This is the 2nd camera oriented and shifted to some other place.
X_cam2 = np.asarray([10, 25, 7]).reshape(3, -1)
R2 = utils.rotate(thetax=90)
t2 = -R2.dot(X_cam2) # t will be R_transpose.cam-centre
t2_norm = t2/np.linalg.norm(t2)

P2 = utils.P_from_krt(K, R2, t2)
pts2d_2 = utils.project(P2, Points)
pts2d_2 = utils.hom_to_euc(pts2d_2)

Ess1, _ = cv2.findEssentialMat(pts2d_1, pts2d_2, K ,cv2.FM_RANSAC)
R2_est1, R2_est2, t2_est = cv2.decomposeEssentialMat(Ess1) # t_est shall be a unit-vector. it wont give the legth, but only scale.

print("R estimated is equal ???  ", (R2-R2_est1).sum(), (R2-R2_est2).sum())
print("Is the t equal ??? ", (t2_norm - t2_est).sum())

#### As you see above, both the `t` as well as `Rs` estimated by opencv are very close-by.


########################################## Case-2 ########################################## 
## Here we shall place two cameras, one with R2, t2 and other with R3, t3
## We shall manually project a set of 3d points on these two cameras.
## we shall do the same things as we did above.
## Except that these R  and t will be the ones wrt the first-camera.


X_cam3 = np.asarray([15, 25, 10]).reshape(3, -1)
R3 = utils.rotate(thetay=68)
t3 = -R3.dot(X_cam3) # t will be R_transpose.cam-centre
t3_norm = t3/np.linalg.norm(t3)

P3 = utils.P_from_krt(K, R3, t3)
pts2d_3 = utils.project(P3, Points)
pts2d_3 = utils.hom_to_euc(pts2d_3)

Ess2, _ = cv2.findEssentialMat(pts2d_2, pts2d_3, K ,cv2.FM_RANSAC)
R3_est1, R3_est2, t3_est = cv2.decomposeEssentialMat(Ess2) # t_est shall be a unit-vector. it wont give the legth, but only direction.


## Note that these R3_est1 or R3_est2 will give you the rotation of Cam3 wrt to Cam2. (assuming that cam2 was at rotation=0).



#### Now let's see how we can get the rotation of Cam3 wrt Cam1(the original origin).
## One way to get this is by findng the essential matrix and then decomposing using pts2d_1 and pts2d_3.

# HOW TO DO THIS ?
# wkt, R2 orients from cam1(origin) to cam2. And R3_est1 orients from cam2 to cam3.
# Then,we can simply orient from cam3 to cam2 using `R3_est1.transpose()`. and then from cam2 to cam1 using (R2_est1.transpose())

# But how do we know if the above steps are correct. Just use essential mat from `pts2d_1` and `pts2d_3` , 
# get the rotation-mat and verify.

Ess3, _ = cv2.findEssentialMat(pts2d_1, pts2d_3, K ,cv2.FM_RANSAC)
R13_est1, R13_est2, t3_est = cv2.decomposeEssentialMat(Ess3) # t_est shall be a unit-vector. it wont give the legth, but only direction.

# lets see in code.

# R_13 means orientation of cam3 wrt cam1. Thinkg why there is .T in the end :-)
R_13 = np.matmul(R2_est1.T, R3_est1.T).T


print((R13_est1 - R_13).sum())
# As we see, the difference between both is of order 10e-8