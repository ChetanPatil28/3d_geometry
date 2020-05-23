import cv2
import numpy as np


## TO-DO

## Grab t1_L, t1_R and t2_L, t2_R, estimate disparity fom both.
## Detect features on t1_L. (fast or shi-tomasi's good features to track or whatever)
## These features must be detected via BUCKETING, unlike traditional non-uniform spatial feature detection. (say 55 features)
## Track where these features land. (all 55 are captured perfectly but some are with more errors)
## Pick only those features who have less error. (discard those with more errors). SO u end up with 25 features(say).
## Now u have 25-correspondences from t1_L and t2_L
## Reproject them back to 3d. (necessary info will be in KITTI's calib.txt)
## U will end up with 25 such 3d-points for both t1_L and t2_L.
## Use these to construct the Consistency matrix.