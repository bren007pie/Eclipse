from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np;

#initialize camera and make a reference to raw capture
camera = PiCamera()
camera.resolution = (640,480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640,480))

#warmup
time.sleep(0.1)

#set up simple blob detector parameters
params = cv2.SimpleBlobDetector_Params()
#change thresholds
params.minThreshold = 100
params.maxThreshold = 255

#Filter by circularity
params.filterByCircularity = True
params.minCircularity = 0.1

#Filter by area
params.filterByArea = True
params.minArea = 1500

#Filter by inertia
params.filterByInertia = True
params.minInertiaRatio = 0.01

#create a drector with the parameters above
ver = (cv2.__version__).split('.')
if int(ver[0]) < 3 :
	detector = cv2.SimpleBlobDetector(params)
else : 
	detector = cv2.SimpleBlobDetector_create(params)

#capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format = "bgr", use_video_port=True):
    #grab the raw NumPy array representing the image
    #then initialize the timestamp
    image = frame.array

    #detect blobs
    keypoints = detector.detect(image)

    #draw detected blobs as red circles
    #cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle
    #corresponds to the size of blob

    image_w_keypoints = cv2.drawKeypoints(image, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    

    #show the frame
    cv2.imshow("Frame with keypoints", image)
    
    key = cv2.waitKey(1) & 0xFF

    #clear the stream (prepare for next frame)
    rawCapture.truncate(0)

    #if the 'q' key was pressed then break the loop
    if key == ord("q"):
        break
    
