import cv2
import numpy as np

def rotate(thetax=0, thetay=0, thetaz=0):
	th_x, th_y, th_z = np.deg2rad(thetax), np.deg2rad(thetay), np.deg2rad(thetaz)
	R, _  = cv2.Rodrigues(np.asarray([th_x, th_y, th_z], dtype=np.float32))
	return R


def project(P_Mat, points_3d):
    ones = np.ones(points_3d.shape[0]).reshape(-1, 1)
    points_3d = euc_to_hom(points_3d)
    points_3d_homo = np.transpose(points_3d, axes=(1, 0) )
    points_2d_homo = np.matmul(P_Mat, points_3d_homo).T # convert to (N, 3) by trnasposing.
    return points_2d_homo

def euc_to_hom(points): # (N, 3 or 4)
	ones = np.ones(points.shape[0]).reshape(-1, 1)
	points_homo = np.concatenate((points, ones), axis = 1)
	return points_homo


def hom_to_euc(points):
	last = points[:, -1].copy()
	points_euc = points/last[:, None]
	return points_euc[:, :-1]


def P_from_krt(K, R, t):
	Rt = np.hstack((R, t))
	P = np.matmul(K, Rt)
	return P

def norm(vec):
	return vec/np.linalg.norm(vec)
