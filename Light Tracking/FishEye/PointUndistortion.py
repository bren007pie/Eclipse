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
    
    round1 = np.round(map1)
    round2 = np.round(map2)
    
    #Location of lightsource in raw/distorted image
    Xpoint = 100#124
    Ypoint = 150#396
    #print(Xpoint)
    
    Xdiff = abs(round1-Xpoint)
    #print(Xdiff)
    
    
    print(Xdiff.min())
    #Xmins is 2 arrays corresponding to the X and Y indices of closest match between Xpoint & Map1 
    Xmins = np.where(Xdiff == Xdiff.min())
    print(Xmins[1][1])
    print(type(Xmins[1]))
    
    #print(round1[92,187])
    #print(round2[92,187])
    
    #list of values in map 2 corresponding to Xmins
    Ypos = round2[Xmins[0],Xmins[1]]
    print(Ypos)
    
    Ydiff = abs(Ypos-Ypoint)
    #index of best match between Ypos and Ypoint
    minidx = np.where(Ydiff == Ydiff.min())
    #print(minidx)
    #print(Ypos[minidx][0])
    
    #Final idices for map1/map2
    FinalX = Xmins[0][minidx][0]
    FinalY = Xmins[1][minidx][0]

    
    print(FinalX,FinalY)
    print(round1[FinalX,FinalY],round2[FinalX,FinalY])
    #print(round1[FinalX,FinalY][0],round2[FinalX,FinalY][0])
    
    cv2.circle(undistorted_img, (Xpoint, Ypoint), 5, (255,0,0), 3)
    cv2.circle(undistorted_img, (FinalX, FinalY), 5, (255,0,0), 3)
    cv2.imshow("bla",undistorted_img)
    
    
    #imshow only works with waitkey fml
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
if __name__ == '__main__':
    for p in sys.argv[1:]:
        undistort(p)
    