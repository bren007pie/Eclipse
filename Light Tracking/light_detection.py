from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2

#initialize camera and make a reference to raw capture
camera = PiCamera()
camera.resolution = (640,480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640,480))

#warmup
time.sleep(0.1)

#choose sigmas for gaussian blur
#must be odd and positive numbers
sigma_x = 11
sigma_y = 11

#set radius for circle showing brightest spot
r = 10

#capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format = "bgr", use_video_port=True):
    # grab the raw NumPy array repreesnting the image
    # then initialize the timestamp
    image = frame.array
    image = cv2.flip(image,0)
    image = cv2.flip(image,1)
    cv2.line(image,(0,240),(640,240),(0,0,255))
    cv2.line(image,(0,235),(640,235),(0,255,0))
    

    #convert image to greyscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #apply Gaussian blurring
    gray = cv2.GaussianBlur(gray, (sigma_x, sigma_y), 0)

    #get the area of the image with the largest intensity value
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)

    #draw circle over brightest pixel
    cv2.circle(gray, maxLoc, r, (255,0,0), 2)
        
    #show the frame with gaussian light detection
    cv2.imshow("Frame", image)
    key = cv2.waitKey(1) & 0xFF

    #clear the stream (prepare for next frame)
    rawCapture.truncate(0)

    #if the 'q' key was pressed then break the loop
    if key == ord("q"):
        break
    

