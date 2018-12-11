#To do:
# - add max size threshold
# - auto brightness threshold selection? 
# - could optimize by not finding contours?????

#all imports needed
from imutils import contours
from imutils.video import FPS
from skimage import measure
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np
import imutils

#choose sigmas for gaussian blur
#must be odd and positive numbers
#too much blurring makes smaller lights harder to see
sigma_x = 7
sigma_y = 7

#set blob brightness threshold
p_thresh = 200

#set blob size threshold
#150 works well with the strong filter
#200 works well without
size_thresh = 50

#resolution
x_res = 640
y_res = 480

#initialize camera and make a reference to raw capture
camera = PiCamera()
camera.resolution = (x_res,y_res)
camera.framerate = 32
camera.exposure_mode = 'off' #'auto'
camera.shutter_speed = 1000 #ranges from 150 to 9000000.
#1000 seems like a good value for bright light detection (flashlight). 

rawCapture = PiRGBArray(camera, size=(x_res,y_res))

#warmup
time.sleep(2)
fps = FPS().start()

#capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format = "bgr", use_video_port=True):
    # grab the raw NumPy array repreesnting the image
    # then initialize the timestamp
    image = frame.array

    #convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #apply Gaussian blurring
    blurred = cv2.GaussianBlur(gray, (sigma_x, sigma_y), 0)

    #apply thresholding to reveal light regions
    #any pixel value >p_thresh will be set to 255
    thresh = cv2.threshold(blurred, p_thresh, 255, cv2.THRESH_BINARY)[1]

    #Perform erosion and dilation to further get rid of noise.
    #Note that excess erosion leads to loss of source detection.
    #In theory can make up for erosion with dilation as long as information is not
    #completely eroded away.
    thresh = cv2.erode(thresh, None, iterations = 1) 
    thresh = cv2.dilate(thresh, None, iterations = 3) 

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
        cv2.circle(image, (int(cX), int(cY)), int(radius), (0,0,255), 3)

    #show the frame with thresholding and multiple lights detected
    cv2.imshow("Frame", image)
    #for debuggin purposes - show the processed frame
    #cv2.imshow("debug", thresh)
    key = cv2.waitKey(1) & 0xFF

    #clear the stream (prepare for next frame)
    rawCapture.truncate(0)

    #if the 'q' key was pressed then break the loop
    if key == ord("q"):
        break

    fps.update()

#output fps info
fps.stop()
print("[INFO] approx FPS: {:.2f}".format(fps.fps()))
#cleanup
cv2.destroyAllWindows()

