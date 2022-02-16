# Undistort testing

import argparse
import cv2
import numpy as np
import sys

DIM=(2560, 1440)
DIM=(1920, 1080)
K=np.array([[781.3524863867165, 0.0, 794.7118000552183], [0.0, 779.5071163774452, 561.3314451453386], [0.0, 0.0, 1.0]])
D=np.array([[-0.042595202508066574], [0.031307765215775184], [-0.04104704724832258], [0.015343014605793324]])

DIM=(4000, 3000)
K=np.array([[1752.5750211793984, 0.0, 1995.612035536046], [0.0, 1753.5569669460147, 1504.879958465866], [0.0, 0.0, 1.0]])
D=np.array([[0.049053553239689844], [0.007995224354946921], [0.003191645723432522], [-0.002814008171914889]])

# K and D hella need adjustment. Also, it appears to be cutting off the right side of the image. To fix once we have better K and D values.

def undistort(img_path):
    img = cv2.imread(img_path)
    h,w = img.shape[:2]
    newMat, ROI = cv2.getOptimalNewCameraMatrix(K, D, DIM, alpha = 1, centerPrincipalPoint = 1)
    # map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, None, newMat, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    cv2.imshow("undistorted", undistorted_img)
    cv2.imwrite(img_path + "_undistort.jpg", undistorted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__ == '__main__':
    for p in sys.argv[1:]:
        undistort(p)