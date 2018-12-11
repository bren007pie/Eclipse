from threading import Thread
from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2

class PiCamThreaded:
    def __init__(self)#x_res,y_res):
        #initialize camera stream
        rawCapture = PiRGBArray(camera, size = (640,480))
        self.stream = camera.capture_continuous(rawCapture,format = "bgr", use_video_port=True)
        #read first frame from stream
        (self.grabbed, self.frame) = self.stream.read()
        #initialize 'stopped' variable
        self.stopped = False

    def start(self):
        #start the thread
        Thread(target = self.update)
        return self

    def update(self):
        #loop until thread is stopped
        while True:
            if self.stopped:
                return
            #read next frame from stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        #return most recent frame that was read
        return self.frame

    def stop(self):
        self.stopped = True
