import sys
import cv2
assert cv2.__version__[0] == '3', 'The fisheye module requires opencv version >= 3.0.0'
import numpy as np


img_path='/home/pi/Eclipse/FishEye'

DIM=(640, 480)
K=np.array([[324.51232784667053, 0.0, 319.01049337166336], [0.0, 324.2725663444019, 246.13206271014946], [0.0, 0.0, 1.0]])
D=np.array([[0.10556938391990513], [-0.032641409080109124], [-0.16439858533661475], [0.10426003475646951]])


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
    
    
    
    
    Xref = (np.round(map1)).astype(int)
    Yref = (np.round(map2)).astype(int)
    
    proof = np.uint8(np.zeros((int(480),int(640))))
   
    a = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    b = np.uint8(a)
    
    for i in range(480):
        for j in range(640):
            Xsample = Yref[i,j]
            Ysample = Xref[i,j]
            
            if Xsample < 0 or Xsample > 479:
                proof[i,j] = 0
                
            
            
            elif Ysample < 0 or Ysample > 639:
                proof[i,j] = 0
                    
            else:
                proof[i,j] = b[Xsample,Ysample]
                #print(proof[i,j])          
            
    
    #cv2.imshow("undistorted", undistorted_img)
    #print("map1", Xref[:,int(1572/2)])
    #print(len(map1[767,:]))
    #cv2.imshow("Xshift", Xshift/30)
    cv2.imshow("plsplsplspls", proof)
    
    
    
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__ == '__main__':
    for p in sys.argv[1:]:
        undistort(p)