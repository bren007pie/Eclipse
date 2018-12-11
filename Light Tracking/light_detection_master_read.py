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
import math
import sys

#change system path to be able to import sunlight sensor driver
#sys.path.append('./SunIOT/SDL_Pi_SI1145');
#import SDL_Pi_SI1145

#define sunlight sensor and function
#sensor = SDL_Pi_SI1145.SDL_Pi_SI1145()
def readSunLight():        
	#time.sleep(0.5)
	vis = sensor.readVisible()
	IR = sensor.readIR()
	UV = sensor.readUV()
	uvIndex = UV / 100.0 #currently not outputting
	return (vis,IR,UV)


#Define threaded Pi Cam class
class PiCamThreaded:
    def __init__(self, resolution = (640,480),framerate = 16):
        #initialize camera and stream
        self.camera = PiCamera()
        self.camera.resolution = resolution
        self.camera.framerate = framerate
        self.camera.exposure_mode = 'off' #'auto'
        self.camera.shutter_speed = 1000
        #self.camera.iso = 100 #try this low iso
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

    def changeShutter(self, exposure):
        self.camera.shutter_speed = exposure

    def stop(self):
        self.stopped = True


#set blob brightness threshold
p_thresh = 150

#set blob size threshold
size_thresh = 5

#resolution
x_res = 640
y_res = 480

#set up serial port
ser = serial.Serial(
    port='/dev/serial0',
    baudrate = 9600, #max baudrate
    parity = serial.PARITY_NONE,#not ensuring accurate transmission
    stopbits = serial.STOPBITS_ONE,
    bytesize = serial.EIGHTBITS,
    timeout = 1
    )

#define lens characteristics (trained) for fisheye undistortion
DIM=(640, 480)
K=np.array([[324.51232784667053, 0.0, 319.01049337166336], [0.0, 324.2725663444019, 246.13206271014946], [0.0, 0.0, 1.0]])
D=np.array([[0.10556938391990513], [-0.032641409080109124], [-0.16439858533661475], [0.10426003475646951]])

#this function uses the lens characteristics to make the undistortion mapping (map1,map2)
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
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, dim3, cv2.CV_32FC1)
    return(map1, map2)

#define location of camera with respect to the windshield in full frame
width = 140 #total width of windshield
height = 70 #total height of windshield
cam_perp_dist = 60 #perpendicular distance from camera to windshield in cm
cam_spacing = 10
cam_x_dist = width/2 - cam_spacing/2 #horizontal distance from left edge of windshield to camera
#^ 65 cm corresponds to the left camera with a spacing of 10 cm between the two. 
cam_y_dist = height/2 #camera should be centred vertically accoutning for the incline of the windshield
x_centreline = width/2 #half of windshield width
FOV = 2*math.atan((width-cam_x_dist)/cam_perp_dist)*180/math.pi #total field of view of the camera in radians
driver_dist = 40 #horizontal distance from driver to left camera
driver_h = 10 #vertical distance (height) of driver above cameras

#start both objects from other Classes
vs = PiCamThreaded().start()
fps = FPS().start()

#warmup
time.sleep(2.0)

#grab one frame and call prep_undistort to produce the mapping
frame = vs.read()
(map1, map2) = prep_undistort(frame)

#round the numbers in map1,map2 to allow for inverse mapping/undistortion
round1 = np.round(map1)
round2 = np.round(map2)

#loop until user quits
#increment = 0
while True:
    #update sunlight sensor inputs (vis, IR, UV)
    #(vis,IR,UV) = readSunLight()
    #test prints
    #print('Vis:' + str(vis))
    #print('IR:' + str(IR))
    #print('UV:' + str(UV))

          
    #set camera exposure according to sensor inputs
    #increment = increment + 1
    #if IR < 300:
    #    vs.changeShutter(int(500*math.pow(1.1,increment)))
    #elif IR < 600:
        #vs.changeShutter(500)
    #don't need an else case because exposure should be set to 500 as IR
    #climbs past 600
        
    #grab frame
    frame = vs.read()#already an array

    #flip in x and y
    frame = cv2.flip(frame,0)
    frame = cv2.flip(frame,1)
    
    #convert image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
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
    labels = measure.label(thresh, neighbors = 8, background = 0) #consider using
    #the additional parameter "connectivity = " which specifies the max. number of
    #orthogonal hops to considera  pixel as a neighbour. Setting this would mean
    #light sources have a max size, and this could also improve speed. 
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

    #reset lists to store alpha,phi for multiple lights.
    alpha = []
    phi = []
    #loop over the contours
    for(i, c) in enumerate(conts):
        #draw bright spot on the image
        #(x, y, w, h) = cv2.boundingRect(c)
        ((cX,cY), radius) = cv2.minEnclosingCircle(c)
        
        #Location of lightsource in raw/distorted image
        Xpoint = cX
        Ypoint = cY        

        #Take difference between rounded map1 and Xpoint
        Xdiff = abs(round1-Xpoint)
                
        #Xmins is 2 arrays corresponding to the X and Y indices of closest match between Xpoint & Map1 
        Xmins = np.where(Xdiff <= Xdiff.min() + 1)        
               
        #list of values in map 2 corresponding to Xmins
        Ypos = round2[Xmins[0],Xmins[1]]
        
        #Take the difference between the list of matched Y positions and Ypoint
        Ydiff = abs(Ypos-Ypoint)
        
        #Get the index of best match between Ypos and Ypoint
        minidx = np.where(Ydiff == Ydiff.min())
              
        #Final indices for map1/map2
        #These are the final undistorted x,y values of the light source. 
        FinalX = Xmins[1][minidx][0]
        FinalY = Xmins[0][minidx][0] #Y is inverted in terms of a normal x-y plane
        #print FinalX, FinalY

        #Convert the pixel locations of light sources to angle with respect to the camera's normal
        #position on windshield
        x_windshield = 1.0*FinalX/x_res*width
        y_windshield = 1.0*(y_res-FinalY)/y_res*height #in the process uninvert y

        #Also convert angles to degrees
        alpha.append(math.atan((x_windshield-cam_x_dist)/cam_perp_dist)*180/math.pi)
        phi.append(math.atan((y_windshield-cam_y_dist)/cam_perp_dist)*180/math.pi)
        
        #print angles for current light
        print('alpha ' + str(i+1) + ': ' + str(alpha[i]))
        print('phi: ' + str(i+1) + ': ' + str(phi[i]))
        print('FOV: ' + str(FOV))
        
        #for getting the angle with respect to the driver using data from both cameras,
        #need to make sure that the two pairs of angles being worked are the right match.
        #Can do this through either master-slave config or simple comparisons. 
        
    #show the frame for debugging
    #for true calibration this would need to be 'unfisheyed'
    cv2.imshow("Frame", frame)
    
    key = cv2.waitKey(1) & 0xFF
    #if the 'q' key was pressed then break the loop
    if key == ord("q"):
        break

    #reset previously read alpha,phi
    alpha_other = 0
    phi_other = 0
    #perform serial read operations if something is in the buffer
    if ser.inWaiting():
        x_array = []
        y_array = []
        for i in range(2):
            x_array.append(ser.read()) 

        for i in range(2):
            y_array.append(ser.read())
        
        x_packed = x_array[0] + x_array[1]
        x_unpacked = struct.unpack('h', x_packed)
        x_read = x_unpacked[0]

        y_packed = y_array[0] + y_array[1]
        y_unpacked = struct.unpack('h', y_packed)
        y_read = y_unpacked[0]
        #print('other x: ' + str(x_read))
        #print('other y: ' + str(y_read))

        #convert to distance on windshield
        x_windshield_other = 1.0*x_read/x_res*width
        y_windshield_other = 1.0*(y_res-y_read)/y_res*height #in the process uninvert y
        #now convert read x,y to angles
        #note addition of cam_spacing to get correct alpha of right camera
        alpha_other = math.atan((x_windshield_other-(cam_x_dist + cam_spacing))/cam_perp_dist)*180/math.pi
        phi_other = math.atan((y_windshield_other-cam_y_dist)/cam_perp_dist)*180/math.pi
        print('other alpha: ' + str(alpha_other))
        print('other phi: ' + str(phi_other))

    #parallax correction math
    #only works for a single light for now
    #naming conventions from module report
    #actual trig is slightly modified
    
    if len(alpha) == 1 and len(phi) == 1 and alpha_other and phi_other:
        #horizontal part:
        theta_c1 = alpha[0]*math.pi/180 #convert each to radians
        theta_c2 = (180-alpha_other)*math.pi/180
        theta_SC = math.pi-theta_c1-theta_c2
        d_c1 = cam_spacing/math.sin(theta_SC)*math.sin(theta_c2)
        d = d_c1*math.sin(theta_c1) #the key perp distance to light source
        #now get the horizontal driver angle and convert to degrees
        d_D = math.sqrt(math.pow(d_c1,2) + math.pow(driver_dist,2)-2*d_c1*driver_dist*math.cos(math.pi-theta_c1))  
        alpha_D = math.asin(d_c1/d_D*math.sin(math.pi-theta_c1))*180/math.pi
    
        #vertical part:
        phi_c = (phi[0] + phi_other)/2.0*math.pi/180 #take the average and convert to radians
        a = d*math.tan(phi_c)
        c = math.sqrt(math.pow(a,2) + math.pow(d,2))
        #now get the vertical driver angle and convert to degrees
        s = math.sqrt(math.pow(c,2) + math.pow(driver_h,2) - 2*c*driver_h*math.cos(math.pi-phi_c))
        phi_D = math.pi - math.asin(c/s*math.sin(math.pi-phi_c))*180/math.pi

        print('Driver alpha: ' + str(alpha_D))
        print('Driver phi: ' + str(phi_D))

    #flush the serial buffer
    ser.flush()
        
    #update FPS count
    fps.update()


#output fps info
fps.stop()
print("[INFO] approx FPS: {:.2f}".format(fps.fps()))
#cleanup
cv2.destroyAllWindows()
vs.stop()
