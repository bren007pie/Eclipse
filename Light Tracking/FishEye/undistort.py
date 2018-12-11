# You should replace these 3 lines with the output in calibration step
import sys
import cv2
assert cv2.__version__[0] == '3', 'The fisheye module requires opencv version >= 3.0.0'
import numpy as np

img_path='/home/pi/Eclipse/FishEye/CalibrationPics'

DIM=(1360, 768)
K=np.array([[710.121208234677, 0.0, 666.0377367477058], [0.0, 700.9755733663874, 495.97352035463274], [0.0, 0.0, 1.0]])
D=np.array([[0.010420751893791887], [0.1195936053677366], [-0.2991482771321817], [0.14035597401977168]])
def undistort(img_path):
    img = cv2.imread(img_path)
    h,w = img.shape[:2]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    cv2.imshow("undistorted", undistorted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__ == '__main__':
    for p in sys.argv[1:]:
        undistort(p)
