#To do:
# - add max size threshold
# - auto brightness threshold selection? 
# - could optimize by not finding contours?????

#all imports needed
from __future__ import print_function
from imutils import contours
from imutils.video import FPS
#from imutils.video.pivideostream import PiVideoStream
#don't need to  import imutils FPS and PiVideoStream if modifiying them here
from skimage import measure
from threading import Thread
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import datetime
import cv2
import numpy as np
import imutils
import sys

#define FPS and threaded Pi Cam classes

class PiCamThreaded:
    def __init__(self, resolution,framerate = 32):
        #initialize camera and stream
        self.camera = PiCamera()
        self.camera.resolution = resolution
        self.camera.framerate = framerate
        self.camera.exposure_mode = 'off' #'auto'
        #self.camera.shutter_speed = 500
        self.rawCapture = PiRGBArray(self.camera, size = resolution)
        self.stream = self.camera.capture_continuous(self.rawCapture,format = "bgr", use_video_port=True)
       #initialize frame and stopped variables
        self.frame = None #self.camera.capture(self.rawCapture, format = "bgr")
        self.stopped = False

    def start(self):
        #start the thread
        t = Thread(target = self.update, args=()) #may not need args()
        t.daemon = True
        t.start()
        return self

    def update(self):
        #loop until thread is stopped
        for f in self.stream:
            #grab array of frame
            self.frame = f.array
            #clear the stream in preparation
            self.rawCapture.truncate(0)

            if self.stopped:
                self.stream.close()
                self.rawCapture.close()
                self.camera.close()
                return

    def read(self):
        #return most recent frame that was read
        return self.frame

    def stop(self):
        self.stopped = True

#choose sigmas for gaussian blur
#must be odd and positive numbers
#too much blurring makes smaller lights harder to see
#sigma_x = 5
#sigma_y = 5

#set blob brightness threshold
p_thresh = 200

#set blob size threshold
#150 works well with the strong filter
#200 works well without
size_thresh = 5

#resolution
x_res = 640
y_res = 480

#define fisheye un-distortion things
#lens characteristics (trained)
DIM=(640, 480)
K=np.array([[324.51232784667053, 0.0, 319.01049337166336], [0.0, 324.2725663444019, 246.13206271014946], [0.0, 0.0, 1.0]])
D=np.array([[0.10556938391990513], [-0.032641409080109124], [-0.16439858533661475], [0.10426003475646951]])



def prep_undistort(frame, balance = 1, dim2=None, dim3=None):
    img = frame
    dim1 = img.shape[:2][::-1]  #dim1 is the dimension of input image to un-distort
    assert dim1[0]/dim1[1] == DIM[0]/DIM[1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"
    if not dim2:
        dim2 = dim1
    if not dim3:
        dim3 = dim1
    scaled_K = K * dim1[0] / DIM[0]  # The values of K is to scale with image dimension.
    scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0
    #This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document failed to make this clear!
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, dim2, np.eye(3), balance=balance)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, dim3, cv2.CV_16SC2)
    return(map1, map2)

def undistort(frame, map1, map2):
    img = frame
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_img

#start both objects from other Classes
vs = PiCamThreaded(resolution = (x_res,y_res)).start()
#instantiate FPS object
fps = FPS().start()

#warmup
time.sleep(2.0)

#grab one frame and get the mapping to undistort
frame = vs.read()
(map1, map2) = prep_undistort(frame)

#debug code for saving an image!
#undist_frame = undistort(frame, map1, map2)
#cv2.imwrite('/home/pi/Eclipse/fisheye_undistorted.jpg',undist_frame)

while True:
    #grab frame
    frame = vs.read()#already an array
    
    #convert from fisheye to perspective
    undist_frame = undistort(frame, map1, map2)

    #convert image to grayscale
    gray = cv2.cvtColor(undist_frame, cv2.COLOR_BGR2GRAY)  
    
    #apply Gaussian blurring    
    #blurred = cv2.GaussianBlur(undist_frame, (sigma_x, sigma_y), 0)

    #apply thresholding to reveal light regions
    #any pixel value >p_thresh will be set to 255
    thresh = cv2.threshold(gray, p_thresh, 255, cv2.THRESH_BINARY)[1]

    #Perform erosion and dilation to further get rid of noise.
    #Note that excess erosion leads to loss of source detection.
    #In theory can make up for erosion with dilation as long as information is not
    #completely eroded away.
    thresh = cv2.erode(thresh, None, iterations = 1) 
    thresh = cv2.dilate(thresh, None, iterations = 5) 

    #perform a connected component analysis on the thresholded image
    #then initialize a  mask to store only larger components
    labels = measure.label(thresh, neighbors = 8, background = 0)
    mask = np.zeros(thresh.shape, dtype = "uint8")

    #loop over unique components
    for label in np.unique(labels):
        #if background label, ignore it
        if label == 0:
            continue
        #otherwise construct label mask and count number of pixels
        labelMask = np.zeros(thresh.shape, dtype = "uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)

        #if numPixels is large enough then add it to mask of large blobs
        if numPixels > size_thresh:
            mask = cv2.add(mask, labelMask)

    #find the contours in the mask and sort from left to right
    conts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    conts = conts[0] if imutils.is_cv2() else conts[1]
    #need to check if mask has at least one blob, otherwise shouldn't look for
    #contours or else the program will crash
    if len(conts) >= 1:
        conts = contours.sort_contours(conts)[0]

    #loop over the contours
    for(i, c) in enumerate(conts):
        #draw bright spot on the image
        #(x, y, w, h) = cv2.boundingRect(c)
        ((cX,cY), radius) = cv2.minEnclosingCircle(c)
        cv2.circle(undist_frame, (int(cX), int(cY)), int(radius), (255,0,0), 3)

    #show the frame with thresholding and multiple lights detected
    cv2.imshow("Frame", undist_frame)

    key = cv2.waitKey(1) & 0xFF
    #if the 'q' key was pressed then break the loop
    if key == ord("q"):
        break

    #update FPS count
    fps.update()
 


#output fps info
fps.stop()
print("[INFO] approx FPS: {:.2f}".format(fps.fps()))
#cleanup
cv2.destroyAllWindows()
vs.stop()
