import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.rcParams["backend"] = "TkAgg"
from mpl_toolkits import mplot3d
import utils
from utils import Camera, TrajectoryFromP, TrajectoryFromPnP
np.random.seed(79)

# matplotlib.use('GTK3Agg')

# In this file, we will track the movement of cameras from the origin.

K = np.load("../camMatrix_720p.npy")
dist = np.zeros(shape=5)
pts3d_11 = np.random.randint(11, 50, size=(8, 3)).astype(np.float64) # 8-points

# Q1. Given a set of Project-Matrices, can you find the location of each Cameras wrt World ?

print("##################### TEST-CASE-1 ################# ")

# TEST-1 assume all the cameras have same orientation and world with some translation only.
Cam2 = Camera(K, Rc=utils.rotate(), center=np.asarray([10, 25, 7]).reshape(3, -1))
Cam3 = Camera(K, Rc=utils.rotate(), center=np.asarray([15, 35, 11]).reshape(3, -1))
Cam4 = Camera(K, Rc=utils.rotate(), center=np.asarray([0, 15, 40]).reshape(3, -1))

# From Projection Mat, we can basically transform a point as `seen by world` to a point as `seen by camera`.
# However, we want the point to be `as seen by the world`. Therefore we should go with P_inverse().

# ie The center of Camera will always be (0, 0, 0) as seen by that Camera.
# However, we want to know this center of Camera as seen by the world. That's why it's simply P_inverse().
# Basically, in order to get the cam-location, we just need to do { P_inv() * [0].T } or simply { -R.inv() * t }

# We'll use the trajectory class.
traject1 = TrajectoryFromP()
traject1.track(Cam2)
traject1.track(Cam3)
traject1.track(Cam4)
traject1.get_cam_locations()

print("##################### TEST-CASE-2 ################# ")
# TEST-2 Now the cameras can have different orientation too.
Cam2 = Camera(K, Rc=utils.rotate(thetax=45), center=np.asarray([10, 25, 7]).reshape(3, -1))
Cam3 = Camera(K, Rc=utils.rotate(thetay=169), center=np.asarray([70, 70, 70]).reshape(3, -1))
Cam4 = Camera(K, Rc=utils.rotate(thetaz=90), center=np.asarray([30, 15, 40]).reshape(3, -1))
Cam5 = Camera(K, Rc=utils.rotate(thetaz=90), center=np.asarray([25, 45, 25]).reshape(3, -1))

traject2 = TrajectoryFromP()
traject2.track(Cam2)
traject2.track(Cam3)
traject2.track(Cam4)
traject2.get_cam_locations()

# As we see, both the test cases yielded the same centers as defined in the above code.
# Now, lets move to a bit difficult problem.

# Q2. Given N 3D points, and their corresponding 2D points as seen by different cameras.
# can you get the trajectory in the order of the given points ?
# Following are the points.
pts2d_12 = utils.project(Cam2.P, pts3d_11)
pts2d_13 = utils.project(Cam3.P, pts3d_11)
pts2d_14 = utils.project(Cam4.P, pts3d_11)
pts2d_15 = utils.project(Cam5.P, pts3d_11)


# Ans. You can estimate the R and t from P-n-P.
t3 = TrajectoryFromPnP(K, pts3d_11)
t3.track_points(utils.hom_to_euc(pts2d_12))
t3.track_points(utils.hom_to_euc(pts2d_13))
t3.track_points(utils.hom_to_euc(pts2d_14))
t3.track_points(utils.hom_to_euc(pts2d_15))
t3.get_cam_locations()
cameras = np.asarray(t3.cam_locs)

ax = plt.axes(projection="3d")
ax.set_xlabel('X-AXIS')
ax.set_ylabel('Y-AXIS')
ax.set_zlabel('Z-AXIS')
ax.set_ylim(0, 100)
ax.set_xlim(0, 100)
ax.set_zlim(0, 100)
ax.plot_wireframe(cameras[:, 0], cameras[:, 1], cameras[:, 2], cmap="Greens")
plt.show()

# Q3. Given a set of matching points across different images, can you get the camera trajetory ?
# Ans. This is just like Visual-Odometry. Good thing is u already have the matched sets. The algo is as follows

# For every subsequent image-pair
# calculate the Rs and ts from EssentialMat.
# Get the correct config of R and t using the Chierality test.
# Traingulate two consecutive pairs to get the `r` so that we can estimate the relative scale.
# Use this scale and then keep on iterating from first.





# before that lets see what is recoverPose.


