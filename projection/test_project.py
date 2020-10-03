import cv2
import numpy as np
import utils
np.random.seed(79)
np.set_printoptions(precision=4)

pts = np.random.randint(11, 50, size=(5, 3)) # 5-points
K = np.load("../camMatrix_720p.npy")

X_cam1 = np.asarray([10, 25, 7]).reshape(3, -1)
R1 = utils.rotate(thetax=90)

# then t shall be
t1 = -R1.dot(X_cam1)

print("-Rtranspose.t ie Cam centre is  ", -R1.T.dot(t1)) #getting the x-cam is simply -R.inv().t since t = -R.X_cam

# let's create the 3x4-Projection-matrix 
P1 = utils.P_from_krt(K, R1, t1)
print("Projection-Mat 1 is ", P1)


pts_2d = utils.project(P1, pts)


print("Finally points are ", pts_2d)
    


