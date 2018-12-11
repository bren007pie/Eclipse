from threading import Thread
from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2

class PiCamThreaded:
    def __init__(self, resolution = (640,480),framerate = 32):
        #initialize camera and stream
        self.camera = PiCamera()
        self.camera.resolution = resolution
        self.camera.framerate = framerate
        self.rawCapture = PiRGBArray(self.camera, size = resolution)
        self.stream = self.camera.capture_continuous(self.rawCapture,format = "bgr", use_video_port=True)
        #initialize frame and stopped variables
        self.frame = None
        self.stopped = False

    def start(self):
        #start the thread
        Thread(target = self.update)
        return self

    def update(self):
        #loop until thread is stopped
        for f in self.stream:
            #grab array of frame
            self.frame = f.array
            #clear the stream in preparation
            self.rawCapture.truncate()
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
