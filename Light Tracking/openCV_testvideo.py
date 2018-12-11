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

#capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format = "bgr", use_video_port=True):
    # grab the raw NumPy array representing the image
    # then initialize the timestamp
    image = frame.array

    #show the frame
    cv2.imshow("Frame", image)
    key = cv2.waitKey(1) & 0xFF

    #clear the stream (prepare for next frame)
    rawCapture.truncate(0)

    #if the 'q' key was pressed then break the loop
    if key == ord("q"):
        break
    
