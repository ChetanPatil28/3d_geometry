import cv2
import numpy as np
import utils
np.random.seed(79)
np.set_printoptions(precision=4)

K = np.load("../camMatrix_720p.npy")

#lets assume there's a Camera at point [10, 25, 7] wrt world
X_cam1 = np.asarray([10, 25, 7]).reshape(3, -1)


# lets also assume Camera orientation aligns with world.



R1 = utils.rotate(thetax=90)

# lets define a few points in the world.
# these three points will be at +4 away in z, +5 away in y and +6 away in x
pts = np.asarray([[10, 25, 11], [10, 30, 7], [16, 25, 7], [10, 25, 7]])

pts = np.random.randint(11, 50, size=(5, 3)) # 5-points


# then t shall be
t1 = -R1.dot(X_cam1)

# extrinsic will be [R | t]
Ext1 = np.hstack((R1, t1))


print("Camera-location is ",X_cam1) 
print("Extrinsic is ", Ext1)
print("-Rtranspose.t is ", -R1.T.dot(t1)) #getting the x-cam is simply -R.inv().t since t = -R.X_cam

# let's create the 3x4-Projection-matrix 
P1 = np.matmul(K, Ext1)
print("Projection-Mat 1 is ", P1)






pts_2d = utils.project(P1, pts)


print("Finally points are ", pts_2d)
    


