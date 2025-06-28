import os
import glob
import cv2
import numpy as np
from sklearn.cluster import KMeans

#set params
test_image_set_idx = 2
cluster_size = 10
T_reduce = 50
T_ini = 0.01
sam_dist = 3
morph_size = 11

# set camera intrinsics
fx = 544.2582548211519
fy = 546.0878823951958
cx = 326.8604521819424
cy = 236.1210149172594
k1 = 0.0369
k2 = -0.0557

#preprocessing utilities
def Depth_preprocess(raw_depth_image):
    for i in range(len(raw_depth_image)):
        raw_depth_image[i] = raw_depth_image[i].astype(np.float64) / 1000  # Convert mm to meters

def pixel2Point3D(i, j, z):
    X = (j - cx) * z / fx
    Y = (i - cy) * z / fy
    Z = z
    return [X, Y, Z]

# load images and depth maps
print(f"Loading image set index: {test_image_set_idx}")
depth_folder = f"tests/test{test_image_set_idx}/depth"
image_folder = f"tests/test{test_image_set_idx}/rgb"

depth_names = sorted(glob.glob(f"{depth_folder}/*.png"))
image_names = sorted(glob.glob(f"{image_folder}/*.png"))

dmaps = [cv2.imread(d, cv2.IMREAD_ANYDEPTH) for d in depth_names]
images = [cv2.imread(i) for i in image_names]

# morphological filtering
for i in range(2):
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_size, morph_size))
    dmaps[i] = cv2.morphologyEx(dmaps[i], cv2.MORPH_CLOSE, k)

Depth_preprocess(dmaps)

# undistort images
camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
for i, distorted_image in enumerate(images):
    images[i] = cv2.undistort(distorted_image, camera_matrix, np.array([k1, k2, 0, 0]))

# convert to 3d points
depth_points = []
depth_pixel = []
image_height, image_width, _ = images[0].shape

for i in range(image_height):
    for j in range(image_width):
        if dmaps[0][i, j] != 0:
            depth_points.append(pixel2Point3D(i, j, dmaps[0][i, j]))
            depth_pixel.append([i, j])

# simple k-means clustering
kmeans = KMeans(n_clusters=cluster_size, random_state=0).fit(np.array(depth_points))
result_image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
cluster_idx_image = np.full((image_height, image_width), -1)
random_colors = [np.random.randint(0, 255, 3).tolist() for _ in range(cluster_size)]

for i, point in enumerate(depth_pixel):
    color_idx = kmeans.labels_[i]
    result_image[point[0], point[1]] = random_colors[color_idx]
    cluster_idx_image[point[0], point[1]] = color_idx

# orb feature extraction
source = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)
target = cv2.cvtColor(images[1], cv2.COLOR_BGR2GRAY)
orb = cv2.ORB_create(nfeatures=10000)
kp_source, des_source = orb.detectAndCompute(source, None)
kp_target, des_target = orb.detectAndCompute(target, None)

# match features
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des_source, des_target)
matches = sorted(matches, key=lambda x: x.distance)

pts_source = [kp_source[m.queryIdx].pt for m in matches]
pts_target = [kp_target[m.trainIdx].pt for m in matches]
pts_source = np.float64(pts_source)
pts_target = np.float64(pts_target)

# fundamental matrix estimation
F, _ = cv2.findFundamentalMat(pts_source, pts_target, cv2.FM_LMEDS)
feature_image = result_image.copy()

for pt in pts_source:
    feature_image = cv2.circle(feature_image, (int(pt[0]), int(pt[1])), 3, (0, 0, 255), -1)

inlier_image = result_image.copy()
inlier_pts_source = []
inlier_pts_target = []
N_ini = [0] * cluster_size
N_first = [0] * cluster_size
N_second = [0] * cluster_size

# inlier detection
for i in range(len(pts_source)):
    pt_src = np.array([*pts_source[i], 1])
    pt_tgt = np.array([*pts_target[i], 1])
    dist = cv2.sampsonDistance(pt_src, pt_tgt, F)
    pt_x, pt_y = map(int, pts_source[i])

    if cluster_idx_image[pt_y, pt_x] != -1:
        N_ini[cluster_idx_image[pt_y, pt_x]] += 1

    if dist < sam_dist:
        inlier_image = cv2.circle(inlier_image, (pt_x, pt_y), 3, (0, 0, 255), -1)
        inlier_pts_source.append(pts_source[i])
        inlier_pts_target.append(pts_target[i])
        if cluster_idx_image[pt_y, pt_x] != -1:
            N_first[cluster_idx_image[pt_y, pt_x]] += 1
    else:
        inlier_image = cv2.circle(inlier_image, (pt_x, pt_y), 3, (255, 0, 0), -1)

# second inlier pass
inlier_pts_source = np.float64(inlier_pts_source)
inlier_pts_target = np.float64(inlier_pts_target)
inlier_image_second = result_image.copy()

F_inlier, _ = cv2.findFundamentalMat(inlier_pts_source, inlier_pts_target, cv2.FM_LMEDS)

for i in range(len(inlier_pts_source)):
    pt_src = np.array([*inlier_pts_source[i], 1])
    pt_tgt = np.array([*inlier_pts_target[i], 1])
    dist = cv2.sampsonDistance(pt_src, pt_tgt, F_inlier)
    pt_x, pt_y = map(int, inlier_pts_source[i])

    if dist < sam_dist:
        inlier_image_second = cv2.circle(inlier_image_second, (pt_x, pt_y), 3, (0, 0, 255), -1)
        if cluster_idx_image[pt_y, pt_x] != -1:
            N_second[cluster_idx_image[pt_y, pt_x]] += 1
    else:
        inlier_image_second = cv2.circle(inlier_image_second, (pt_x, pt_y), 3, (0, 255, 0), -1)

# dyname region detection
sum_inlier = sum(N_ini)
sum_second = sum(N_second)
dynamic_image = np.full((image_height, image_width, 3), [0, 0, 255], dtype=np.uint8)

for j in range(cluster_size):
    if N_ini[j] == 0:
        continue
    module1 = (N_ini[j] - N_first[j]) / N_ini[j] * 100 > T_reduce
    module2 = N_ini[j] / sum_inlier > T_ini
    module3 = (N_ini[j] / sum_inlier - N_second[j] / sum_second) > 0
    if module1 and module2 and module3:
        dynamic_image[cluster_idx_image == j] = [0, 255, 0]

# visualisation
row1 = np.hstack((images[0], result_image, feature_image))
row2 = np.hstack((inlier_image, inlier_image_second, dynamic_image))
full_image = np.vstack((row1, row2))

cv2.imshow("Result Image", full_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
