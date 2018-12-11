#To do:
# - add max size threshold
# - auto brightness threshold selection? 
# - could optimize by not finding contours?????

#all imports needed
from __future__ import print_function
from imutils import contours
from imutils.video import FPS
from skimage import measure
from threading import Thread
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import datetime
import cv2
import numpy as np
import imutils
import serial 
import struct

#define FPS and threaded Pi Cam classes

class PiCamThreaded:
    def __init__(self, resolution = (640,480),framerate = 32):
        #initialize camera and stream
        self.camera = PiCamera()
        self.camera.resolution = resolution
        self.camera.framerate = framerate
        self.camera.exposure_mode = 'off' #'auto'
        #self.camera.shutter_speed = 100000
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
#sigma_x = 3
#sigma_y = 3

#set blob brightness threshold
p_thresh = 200

#set blob size threshold
#150 works well with the strong filter
#200 works well without
size_thresh = 50

#resolution
x_res = 640
y_res = 480

#set up serial communication between pi's
ser = serial.Serial(
    port='/dev/serial0',
    baudrate = 9600, #max baudrate
    parity = serial.PARITY_NONE,#not ensuring accurate transmission
    stopbits = serial.STOPBITS_ONE,
    bytesize = serial.EIGHTBITS,
    timeout = 1
    )

#multithreading of serial read
#def readFromPort(ser):
#    if ser.inWaiting():
#        #read ints of x,y, coordinates from other pi    
#        x_packed = ser.readline()
        #x_packed = ser.read()
        #print(x_packed)
        #x_packed_slice = x_packed[:2]
        #print(x_packed_slice)
        #x_unpacked = struct.unpack('h',x_packed_slice)
        #x = x_unpacked[0]
        #print(x)

#t2 = Thread(target = readFromPort, args = (ser))
#t2.daemon = True
#t2.start()

#start both objects from other Classes

vs = PiCamThreaded().start()
fps = FPS().start()

#warmup
time.sleep(2.0)
#loop until user quits
#for i in range(200):

while True:
    #grab frame
    frame = vs.read()#already an array
    #convert image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #apply Gaussian blurring
    #blurred = cv2.GaussianBlur(gray, (sigma_x, sigma_y), 0)

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
        cv2.circle(frame, (int(cX), int(cY)), int(radius), (0,0,255), 3)
        
    #show the frame with thresholding and multiple lights detected
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF
    #if the 'q' key was pressed then break the loop
    if key == ord("q"):
        break

    #prepare serial buffer (may not need this)
 
   
    #check if the serial buffer is processing incoming bytes
    #(shouldn't attempt to read if no bytes are being handled)

    #ser.close()
    #if ser.isOpen() == False:
    #   ser.open()
        #time.sleep(0.1)
    #ser.flush()
    if ser.inWaiting():
        x_array = []
        y_array = []
        for i in range(2):
            x_array.append(ser.read()) 
            #print(x_array[i])
        for i in range(2):
            y_array.append(ser.read())
            #print(y_array[i])
        
        x_packed = x_array[0] + x_array[1]
        x_unpacked = struct.unpack('h', x_packed)
        x = x_unpacked[0]

        y_packed = y_array[0] + y_array[1]
        y_unpacked = struct.unpack('h', y_packed)
        y = y_unpacked[0]
        print('x: ' + str(x))
        print('y: ' + str(y))
    #if ser.inWaiting():
    #    #time.sleep(0.5)
    #    #read ints of x,y, coordinates from other pi    
    #    x_packed = ser.readline()     
    #    print(x_packed)
    #    x_packed_slice = x_packed[:2]
    #    print(x_packed_slice)
    #    x_unpacked = struct.unpack('h',x_packed_slice)
    #    x = x_unpacked[0]        
    #    print(x)
    #check length to make sure there's no error
    #if len(x_packed)==4 :
    #    #slice the byte that was read in two, because it's doubled up for some reason
    #    x_packed_slice = x_packed[:2]
    #    print(x_packed_slice)
    #    x_unpacked = struct.unpack('h',x_packed_slice)
    #    x = x_unpacked[0]
    #    print(x)
    #    x_unpacked = struct.unpack('h',x_packed)
    #    x = x_unpacked[0]
    #    print(x)
    ser.flush()

        

        
    #update FPS count
    fps.update()

 


#output fps info
fps.stop()
print("[INFO] approx FPS: {:.2f}".format(fps.fps()))
#cleanup
cv2.destroyAllWindows()
vs.stop()
