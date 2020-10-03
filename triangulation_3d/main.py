import cv2
import os

pth = "../Imgs_3D"
files = os.listdir(pth)
files = [i for i in files if i.lower().endswith("jpg")]
print(files)

# def get_keypoints_per_img(img):
#     pass


# for f in files:
#     img = cv2.imread(os.path.join(pth, f))
#     # print("Img shape is ", img.shape)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     # sift = cv2.SIFT_create()
#     sift = cv2.xfeatures2d_SIFT.create()
#     kp, desc = sift.detectAndCompute(gray, None)
#     sample_pnt = kp[0]
#     sample_desc = desc[0]
#     # print("samle desc shape is ", sample_desc.shape)
#     # print("Kp shape is ", sample_pnt.pt, sample_pnt.angle, sample_pnt.size, sample_pnt.response)
#     img_draw = cv2.drawKeypoints(gray, kp, img)
#     cv2.imshow("img", img)
#     cv2.waitKey(0)
# cv2.destroyAllWindows()

img1 = cv2.imread(os.path.join(pth, files[1]), cv2.IMREAD_GRAYSCALE)          # queryImage
img2 = cv2.imread(os.path.join(pth, files[5]), cv2.IMREAD_GRAYSCALE) # trainImage
# Initiate SIFT detector
sift = cv2.xfeatures2d_SIFT.create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
# BFMatcher with default params
bf = cv2.BFMatcher()
print("k1 and k2 shapes are ", len(kp1), len(kp2))

matches = bf.knnMatch(des1,des2,k=2)

print('matches shaoe ', len(matches))
# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
# cv.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imwrite("../matched.png ", img3)


