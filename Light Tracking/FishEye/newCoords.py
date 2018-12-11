import sys
import cv2
assert cv2.__version__[0] == '3', 'The fisheye module requires opencv version >= 3.0.0'
import numpy as np


img_path='/home/pi/Eclipse/FishEye'

DIM=(1360, 768)
K=np.array([[710.121208234677, 0.0, 666.0377367477058], [0.0, 700.9755733663874, 495.97352035463274], [0.0, 0.0, 1.0]])
D=np.array([[0.010420751893791887], [0.1195936053677366], [-0.2991482771321817], [0.14035597401977168]])

def undistort(img_path, balance=1, dim2=None, dim3=None):
    img = cv2.imread(img_path)
    dim1 = img.shape[:2][::-1]  #dim1 is the dimension of input image to un-distort
    assert dim1[0]/dim1[1] == DIM[0]/DIM[1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"
    if not dim2:
        dim2 = dim1
    if not dim3:
        dim3 = dim1
    scaled_K = K * dim1[0] / DIM[0]  # The values of K is to scale with image dimension.
    scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0
    # This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document failed to make this clear!
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, dim2, np.eye(3), balance=balance)
    #map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, dim3, cv2.CV_16SC2)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, dim3, cv2.CV_32FC1)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    Xnew = np.round(map1)
    Ynew = np.round(map2)
    
    
    Xshift = Xnew - np.min(Xnew)
    Yshift = Ynew - np.min(Ynew)
    print(Xshift.shape)
    print(Yshift.shape)
    
    print("map1 min", np.max(Xshift))
    print("sdgsdg", np.max(Yshift))
    
    proof = np.zeros((int(1230),int(1578)))
    print(type(Xnew))
    
    for i in range(int(768)):
        for j in range(int(1360)):
            Xboy = int(Xshift[i,j])
            Yboy = int(Yshift[i,j])
            proof[Yboy,Xboy] = 255
    
    #cv2.imshow("undistorted", undistorted_img)
    print("map1", Xnew[:,int(1572/2)])
    #print(len(map1[767,:]))
    #cv2.imshow("Xshift", Xshift/30)
    cv2.imshow("plsplsplspls", proof)
    
    
    
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__ == '__main__':
    for p in sys.argv[1:]:
        undistort(p)
