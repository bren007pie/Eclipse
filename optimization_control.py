from imutils import contours
from imutils.video import FPS
from skimage import measure
from threading import Thread
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import datetime
import numpy as np
import imutils
import serial 
import struct
import math
import sys
import cv2
import os
import RPi.GPIO as GPIO
import subprocess

##THIS VERSION OF THE INTEGRATED CODE HAS GRID ADDRESSING IN A SEPARATE THREAD, FOR A TOTAL OF 3 THREADS##

subprocess.call("./pre_eclipse.sh", shell=True)

#change system path to be able to import sunlight sensor driver
#sys.path.append('./SunIOT/SDL_Pi_SI1145');
#import SDL_Pi_SI1145

#define sunlight sensor and function
#sensor = SDL_Pi_SI1145.SDL_Pi_SI1145()
#def readSunLight():        
#	#time.sleep(0.5)
#	vis = sensor.readVisible()
#	IR = sensor.readIR()
#	UV = sensor.readUV()
#	uvIndex = UV / 100.0 #currently not outputting
#	return (vis,IR,UV)


class PiCamThreaded:
    def __init__(self, resolution = (640,480),framerate = 16):
        #initialize camera and stream
        self.camera = PiCamera()
        self.camera.resolution = resolution
        self.camera.framerate = framerate
        self.camera.exposure_mode = 'off' #'auto'
        self.camera.shutter_speed = 250
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


########################################

class GridAddressing:
    
    def __init__(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(0)

        #list of pins to be used as outputs
        channels = list(range(4,14))+list(range(16,20))
        GPIO.setup(channels, GPIO.OUT, initial = 0)

        #list of pins corresponding to row and column voltage control
        self.bottomRowPins = channels[:7]
        self.topRowPins = channels[7:]
            

    def start(self):
        t = Thread(target = self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        global phi_other
        
        prev_x1 = None
        prev_x2 = None
        prev_y1 = None
        prev_y2 = None
        
        while True:
            #constants and variables
            xCam1 = 55
            yCam2 = 50
            dSensorToWindshield = 20
            acceptableError = 3

            #outputs from Light Tracking (in deg)

            if alpha[0] and phi_other[0]:   #serial MUST be connected to work             
                alpha1 = alpha[0]#alpha_other[0] #-10
                phi1 = phi_other[0]#phi[0] #-15#

                alpha2 = alpha[1]#None#60
                if phi_other[1]:
                    phi2 = phi_other[1]#-15 +phi_other[1]#-15 + phi_other[1]#None #0
            else:
                alpha1 = None
                phi1 = None
                alpha2 = None
                phi2 = None

            #outputs from Driver Tracking (in cm)
            #eye_x and eye_y must be mirrored to get the correct
            #displacement from the perspective of the driver
            horizontalDisplacement = -eye_x 
            #print('hD: ' + str(horizontalDisplacement))
            verticalDisplacement = -eye_y
            #print('vD: ' + str(verticalDisplacement))
            depthSensorToDriver = d_dist

            x1 = y1 = x2 = y2 = x3 = y3 = x4 = y4 = None
            
            dDriver = dSensorToWindshield + depthSensorToDriver
            xDriver = xCam1 + horizontalDisplacement
            #print('xDriver: ' + str(xDriver))
            yDriver = yCam2 + verticalDisplacement
            #print('yDriver: ' + str(yDriver))
            
            if type(alpha1) != type(None):
                xProjection1 = dDriver*math.tan(math.radians(alpha1)) + xDriver
                yProjection1 = dDriver*math.tan(math.radians(phi1)) + yDriver
                
                column1 = int(xProjection1/20)
                row1 = int(yProjection1/20)
                
                if (column1 >= 0 and column1 < 7) and (row1 == 1 or row1 == 2):
                    #print("blah1")
                    x1 = column1
                    y1 = row1
                
                if type(alpha2) != type(None) and type(phi2) != type(None):
                    xProjection2 = dDriver*math.tan(math.radians(alpha2)) + xDriver
                    yProjection2 = dDriver*math.tan(math.radians(phi2)) + yDriver
                    
                    column2 = int(xProjection2/20)
                    row2 = int(yProjection2/20)
                    
                    if (column2 >= 0 and column2 < 7) and (row2 == 1 or row2 == 2):
                        #print("blah2")
                        x2 = column2
                        y2 = row2
                     
            #begin actual grid addressing
            if type(x1) != type(None):
                if y1 == 1:
                    #if location changed in y or x, make previous transparent
                    if prev_x1 != x1 and type(prev_x1) != type(None):
                        GPIO.output(self.bottomRowPins[prev_x1],0)
                    if prev_y1 != y1 and type(prev_y1) != type(None):
                        GPIO.output(self.topRowPins[prev_x1],0)
                    #make new one opaque
                    GPIO.output(self.bottomRowPins[x1],1)
                elif y1 == 2:                   
                    if prev_x1 != x1 and type(prev_x1) != type(None):
                        GPIO.output(self.topRowPins[prev_x1],0)
                    if prev_y1 != y1 and type(prev_y1) != type(None):
                        GPIO.output(self.bottomRowPins[prev_x1],0)
                        
                    GPIO.output(self.topRowPins[x1],1)
                    
                if type(x2) != type(None): #second light!
                    if y2 == 1:
                        if prev_x2 != x2 and type(prev_x2) != type(None):
                            GPIO.output(self.bottomRowPins[prev_x2],0)
                        if prev_y2 != y2 and type(prev_y2) != type(None):
                            GPIO.output(self.topRowPins[prev_x2],0)

                        GPIO.output(self.bottomRowPins[x2],1)
                      
                    elif y2 == 2:
                        if prev_x2 != x2 and type(prev_x2) != type(None):
                            GPIO.output(self.topRowPins[prev_x2],0)
                        if prev_y2 != y2 and type(prev_y2) != type(None):
                            GPIO.output(self.bottomRowPins[prev_x2],0)

                        GPIO.output(self.topRowPins[x2],1)

                #else:
                #    print('ha ha')
                #    if y2 == 1:
                #        GPIO.output(self.bottomRowPins[prev_x2],0)
                #    elif y2 == 2:
                #        GPIO.output(self.topRowPins[prev_x2],0)
                    
            else:
                GPIO.output(self.bottomRowPins,0)
                GPIO.output(self.topRowPins,0)

            #save the most recent value of x1 before it changes
            prev_x1 = x1
            prev_y1 = y1
            prev_x2 = x2
            prev_y2 = y2   
            
        return

    def stop(self):
        self.stopped = True
        GPIO.cleanup()
        
###############################

class HeadTrackingThreaded:

    def __init__(self):        
        cv2.setNumThreads(0)#restrict opencv functions to one thread
        os.chdir("/usr/local/share/OpenCV/haarcascades/")
        self.face_cascade = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_eye.xml')
        self.cap = cv2.VideoCapture("/dev/video0") #sets up video capture, video1 is the webcam, video0 is the picam
        #camera has atributes set by cap.set(cv2.CAP_PROP_EXPOSURE, 40) #Reference: https://docs.opencv.org/trunk/d4/d15/group__videoio__flags__base.html#gaeb8dd9c89c10a5c63c139bf7c4f5704d
        #CAP_PROP_BRIGHTNESS
        #CAP_PROP_CONTRAST
        #CAP_PROP_SATURATION
        #CAP_PROP_GAIN
        #CAP_PROP_AUTO_EXPOSURE ?
        #no exposure, no monochrome, iso speed
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)
        self.campixelwidth = self.cap.get(3) #cv2.CAP_PROP_FRAME_WIDTH
        self.campixelheight = self.cap.get(4)  #cv2.CAP_PROP_FRAME_HEIGHT

        self.ultrasonicsetup(False,40) #True if ultrasonic is hooked up, false if otherwise
        self.eyedistance= []
        self.stopped = False

    def start(self):
        t = Thread(target = self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        global eye_x #NEED THESE OTHERWISE IT DONNNT WORK
        global eye_y
        
        while True:
            ret, img = self.cap.read() #reads in the image
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #converts to grayscale
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5,0,(60, 60)) #detects faces in the grayscale image
            #parameters to detectMultiScale: image, scalefactor (image size reduction), minNeighbours, flags, minSize

            d_dist = self.distance(False) #does the ultrasonic distance measurement, True is the debug
              
            eyecentres = []
            for (x,y,w,h) in faces: #if there are no faces this doesn't run
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),1) #draws the rectangle around the corners in blue to the image object
                cv2.circle(img, (int(x+w/2),int(y+h/2)), 3, (255,0,0),2) #draws a blue circle of radius 3 in the centre of the face to the image object. Pixel reference need to be an integer
                roi_gray = gray[y:int(y+h/2), x:x+w] #defines a new region to search for eyes in the top half of the face
                roi_color = img[y:int(y+h/2), x:x+w]

                facecentre = self.getfacecentre(faces,False) #debug prints out face info
                
                eyes = self.eye_cascade.detectMultiScale(roi_gray) #detects the eyes within the face

                numeyes = self.getnumeyes(eyes,False)      
            
                #cuts the number of eyes down to 2 "randomly"
                while numeyes > 2:
                    eyes = np.delete(eyes,2,0) #deletes the eye at index 2, or the 3rd eye
                    numeyes = int(eyes.size/4)

                eyecentres = self.geteyecentres(eyes,x,y,False)
                self.eyedistance = self.geteyedistance(eyecentres, d_dist, False)
                
                for (ex,ey,ew,eh) in eyes: #writes all the circles
                    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),1)
                    cv2.circle(img, (int(x+ex+ew/2),int(y+ey+eh/2)), 2, (0,255,0),1) #draws a green circle of radius 2 in the centre of the eye to the image object. Pixel reference need to be an integer.
                    #all eye dimensions are with reference to the face so need to add the face coord to each one
                    ###print("Eye Position:",(x+ex+ew/2, y+ey+eh/2,"%.1f cm" % dist)) #prints the middle of the eye location


            cv2.imshow('img',img) #img reference is 0,0 in top left
            key = cv2.waitKey(1) & 0xFF
            #if the 'q' key was pressed then break the loop
            if key == ord("q"):
                break

            #append z-distance to end of eyedist list
            self.eyedistance.append(d_dist)

            #take the average of eye x positions and send driver info
            if len(eyecentres) == 2:
                eyedist = self.geteyedistance(eyecentres,d_dist,True)
                
                eye_x = (eyedist[0][0] + eyedist[1][0])/2
                eye_y = (eyedist[0][1] + eyedist[1][1])/2
                #print('eye_x: ' + str(eye_x))
                #print('eye_y: ' + str(eye_y))
                
            if self.stopped:
                self.cap.release()
                return

        return 
        
    def read(self):
        #output key head tracking information for the main body of the program to use
        return self.eyedistance

    def stop(self):
        self.stopped = True

    def ultrasonicsetup(self,enable,manualdistance): #setups up ultrasonic range finder, if enable is false it doesn't run
        global Uenable, GPIO_TRIGGER, GPIO_ECHO, manualdist  #setups a global variables. Uenable that stops all ultrasonic stuff
        Uenable = enable
        manualdist = manualdistance
        if enable:
            GPIO.setwarnings(False) #This disables the warning about channel being in use
            GPIO.setmode(GPIO.BCM) #GPIO Mode (BOARD or BCM)
            #set GPIO Pins
            GPIO_TRIGGER = 22
            GPIO_ECHO = 27
            ##Pin colours##
            #white = 5V VCC
            #blue = Trigger pin 22 
            #green = ECHO pin 27 Voltage Divider 330, node, then 470 ohms to ground 
            #purple = GND
            #set GPIO direction (IN / OUT)
            GPIO.setup(GPIO_TRIGGER, GPIO.OUT)
            GPIO.setup(GPIO_ECHO, GPIO.IN)
        return

    def distance(self,debug): #gets distance from the ultrasonic range finder
        #manualdist = 50
        if Uenable: #if ultrasonics are on   
            GPIO.output(GPIO_TRIGGER, True) # set Trigger to HIGH
            time.sleep(0.00001) # set Trigger after 0.01ms to LOW
            GPIO.output(GPIO_TRIGGER, False)
            StartTime = time.time()#setup up the time objects
            StopTime = time.time()
            while GPIO.input(GPIO_ECHO) == 0: # save StartTime, these while loops make it pause if not hooked up
                StartTime = time.time()
            while GPIO.input(GPIO_ECHO) == 1: # save time of arrival
                StopTime = time.time()
            TimeElapsed = StopTime - StartTime # time difference between start and arrival
            distance = (TimeElapsed * 34300) / 2 # multiply with the sonic speed (34300 cm/s) and divide by 2, because there and back
        elif (Uenable == False) and (manualdist == 0):
            distance = -1 #-1 used to indicate ultrasonic sensor is not on or can fill with a set value in cm can still work
        else:
            distance = manualdist
        
        if debug:
            print("Distance:\n", distance)
        return distance
   
    def waitfor1face(self,faces): #limiting to 1 face makes it infinite loop for some reason, 
        numfaces = int(faces.size/4)
        while numfaces > 1: #stops the program until there is only 1 face in the frame
            print("Please try to make there only be 1 face")
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            numfaces = int(faces.size/4)
        return

    def getfacecentre(self,faces,debug): #gets centre of the face and returns it as an x and y list
        try: #error checking if there were no faces detected
            centres = [float(faces[0][0] + faces[0][2]/2), float(faces[0][1]+ faces[0][3]/2)] #gets x and y of face centre
            if debug:
                print("Face array:\n", faces) #prints the contents of the faces arrays
                #print("Number faces:\n", numfaces) #not a thing anymore
                print("Face centre:\n", centres)
        except IndexError:
            centres=[[0,0]]
            print("No faces detected! Set to 0,0")

        return centres

    def geteyecentres(self,eyes,x,y, debug): #Getting the Eye Centres makes an array (should just make this a function)
        
        centres = [] #empty list, defined each time
        for (ex,ey,ew,eh) in eyes:
            centres.append([x+ex+ew/2, y+ey+eh/2])
        if debug:
            print("EyeCentres:\n", centres)

        return centres

    def getnumeyes(self,eyes,debug):
        try: #error checking if there is no eyes detected
            num = int(eyes.size/4) #finds the number of eyes, is a numpy.ndarray, getting the number of ellements and dividing by 4
            if debug:
                print(num, " eyes detected")
        except AttributeError:
            num = -1
            print("no faces detected! Set to -1!")
        return num

    def geteyedistance(self,eyecentres, dist, debug):
        #FOV 60 degrees (assume both directions but test) #https://support.logitech.com/en_us/article/17556
        #example eyecentres
        #eyecentres = [[100, 300], [200,300]] #left and right eye
        #dist = 20 #person is 20 cm away
        #resolution = 1280 in x 720 in y
        #FOV 60 degrees (assume both directions but test) #https://support.logitech.com/en_us/article/17556

        eyedist = []

        FOVx = 42.6*math.pi/180 #half the FOV in degrees determined experimentally, converts FOV in radians
        Resolutionx = self.campixelwidth 
        objectplanex = dist*math.tan(FOVx) #gets half the distance of the frame
        scalex = objectplanex*2/Resolutionx #gets cm/pixel at that distance, have to multiply in the start to get the full distance of the frame

        FOVy = 32.1*math.pi/180 #half the FOV in degrees determined experimentall, converts FOV in radians
        Resolutiony = self.campixelheight 
        objectplaney = dist*math.tan(FOVy)
        scaley = objectplaney*2/Resolutiony #gets cm/pixel at that distance

        for (x,y) in eyecentres:
            eyedist.append([scalex*x-objectplanex, scaley*y-objectplaney])
        try:
            diffx = abs(eyedist[0][0] - eyedist[1][0])
            diffy = abs(eyedist[0][1] - eyedist[1][1])
        except IndexError:
            #print("Only 1 eye to clalculate, no difference!")
            diffx = 0
            diffy = 0
        #if debug:
            #print(eyecentres)
            #print("Eye Maps:\n", mapx, mapy)
            #print("Eye Distances:\n",eyedist)
            #print("X difference:\n", diffx)
            #print("Y difference:\n", diffy)
        
        return eyedist


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
    baudrate = 9600, 
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

#This function takes in raw pixel coordinates and undistorts them
def undistort(Xpoint, Ypoint):
    #Take difference between rounded map1 and Xpoint
    Xdiff = abs(round1-Xpoint)
    #Xmins is 2 arrays corresponding to the X and Y indices of closest
    #match between Xpoint & Map1 
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
    return(FinalX,FinalY)


def correctedAngleCalc(pixelX,pixelY):
    pixelXMax = x_res - row_blk_right[pixelY] #in reality the maxes of these will depend on
    pixelYMax = y_res - col_blk_top[pixelX] #the black bars
    cal_depth = 0.45  #whatever depth it takes in calibration to fill the frame
    cal_width = 1 #ex. width of windshield
    cal_height = 0.65 #ex. height of windshield
    alpha = math.degrees(math.atan((cal_width/2)*((pixelX-x_res/2)/(pixelXMax - x_res/2))/cal_depth))
    phi = -math.degrees(math.atan((cal_height/2)*((pixelY-y_res/2)/(pixelYMax - y_res/2))/cal_depth))
    #print(str(alpha))
    #print(str(phi))
    return(alpha,phi)

#Set up GLOBAL alpha, phis for use in grid addressing!
alpha = [None,None]
phi = [None,None]
alpha_other = [None,None] #from other pi
phi_other = [None,None]
eye_x = 0
eye_y = 0
d_dist = 70

#start objects from other Classes
vs = PiCamThreaded().start()
ga = GridAddressing().start()
ht = HeadTrackingThreaded().start()
fps = FPS().start()

#warmup
time.sleep(2.0)

#grab one frame and call prep_undistort to produce the mapping
frame = vs.read()

(map1, map2) = prep_undistort(frame)
#round the numbers in map1,map2 to allow for inverse mapping/undistortion
round1 = np.round(map1)
round2 = np.round(map2)

#generate 'lookup table' for black bar size using map1 and map2
row_blk_left = []
row_blk_right = []
col_blk_bot = []
col_blk_top = []
for i in range(y_res):
    row_blk_left.append(len(np.where(map1[i,:] < 0)[0]))
    row_blk_right.append(len(np.where(map1[i,:] > x_res)[0]))
    
#print(len(np.where(map1[200,:]<0)[0]))
for i in range(x_res):
    col_blk_bot.append(len(np.where(map2[:,i] < 0)[0]))
    col_blk_top.append(len(np.where(map2[:,i] > y_res)[0]))


#loop until user quits
while True:
        
    #grab frame
    frame = vs.read()#already an array

    #flip in x and y
    #frame = cv2.flip(frame,0)
    #frame = cv2.flip(frame,1)
    
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
    #orthogonal hops to consider a pixel as a neighbour. Setting this would mean
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
    #alpha = []
    #phi = []
    #loop over the contours
    for(i, c) in enumerate(conts):
        #draw bright spot on the image
        #(x, y, w, h) = cv2.boundingRect(c)
        ((cX,cY), radius) = cv2.minEnclosingCircle(c)
        
        #Location of lightsource in raw/distorted image
        Xpoint = cX
        Ypoint = cY        

        (FinalX, FinalY) = undistort(cX,cY)               

        #using new angle calc method from calibration
        (temp_alpha,temp_phi) = correctedAngleCalc(FinalX,FinalY)
        alpha[i] = temp_alpha
        phi[i] = temp_phi
        
        #print angles for current light
        #print(str(row_blk_left[int(FinalY)]))
        print('alpha ' + str(i+1) + ': ' + str(alpha[i]))
        print('phi ' + str(i+1) + ': ' + str(phi[i]))
       

    #show the frame for debugging
    #for true calibration this would need to be 'unfisheyed'
    cv2.imshow("Frame", frame)
      
    
    key = cv2.waitKey(1) & 0xFF
    #if the 'q' key was pressed then break the loop
    if key == ord("q"):
        break
       

    #perform serial read operations if something is in the buffer
    if ser.inWaiting():        
        phi1_array = []
             
        for i in range(2):
            phi1_array.append(ser.read())
                      
        phi1_packed = phi1_array[0] + phi1_array[1]
        if len(phi1_packed) == 2:
            phi1_unpacked = struct.unpack('h', phi1_packed)
            phi1_read = phi1_unpacked[0]
            phi_other[0] = phi1_read/100

        print('other phi 1: ' + str(phi_other[0]))           
       
        phi2_array = []
            
        for i in range(2):
            phi2_array.append(ser.read())

        phi2_packed = phi2_array[0] + phi2_array[1]
        if len(phi2_packed) == 2:
            phi2_unpacked = struct.unpack('h', phi2_packed)
            phi2_read = phi2_unpacked[0]           

            if abs(phi2_read-phi1_read) > 100: #if difference > 1 deg
                phi_other[1] = phi2_read/100 #separate phi for separate light        
                print('other phi 2: ' + str(phi_other[1]))
            else:
                phi_other[0] = phi2_read/100 #if too close, update same light
                
    #flush the serial buffer to avoid falling really behind 
    ser.flush()
    ser.reset_input_buffer()
        
    #update FPS count
    fps.update()


#output fps info
fps.stop()
print("[INFO] approx FPS: {:.2f}".format(fps.fps()))
#cleanup
cv2.destroyAllWindows()
vs.stop()
ga.stop()
