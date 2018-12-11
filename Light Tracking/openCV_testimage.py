from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2

#initialize camera and make a reference to the raw capture
camera = PiCamera()
rawCapture = PiRGBArray(camera)

#warmup time
time.sleep(0.1)

#get image
#note BGR format
camera.capture(rawCapture, format = "bgr")
image = rawCapture.array

#display and wait for keypress
cv2.imshow("Image", image)
cv2.waitKey(0)

