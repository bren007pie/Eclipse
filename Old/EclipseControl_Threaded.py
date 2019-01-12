#imports for lighttracking
from __future__ import print_function
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

##THIS VERSION OF THE INTEGRATED CODE HAS HEAD TRACKING IN A SEPARATE THREAD, FOR A TOTAL OF 3 THREADS##

#TODO Clean everything up, don't do try-excepts for no reason, finish multithreading


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


#Define threaded Pi Cam class
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
        while True:
            ret, img = self.cap.read() #reads in the image
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #converts to grayscale
            faces = self.face_cascade.detectMultiScale(gray, 1.6, 5,0,(60, 60)) #detects faces in the grayscale image
            #parameters to detectMultiScale: image, scalefactor (image size reduction), minNeighbours, flags, minSize

            dist = self.distance(False) #does the ultrasonic distance measurement, True is the debug
              

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
                self.eyedistance = self.geteyedistance(eyecentres, dist, False)
                
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
            self.eyedistance.append(dist)

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
            GPIO_TRIGGER = 18
            GPIO_ECHO = 24
            #set GPIO direction (IN / OUT)
            GPIO.setup(GPIO_TRIGGER, GPIO.OUT)
            GPIO.setup(GPIO_ECHO, GPIO.IN)
        return

    def distance(self,debug): #gets distance from the ultrasonic range finder
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

        #My way
        eyedist = []
        FOVx = 42.6*math.pi/180 #FOV in degrees determined experimentall, converts FOV in radians
        thetax = FOVx/2
        Resolutionx = self.campixelwidth 
        objectplanex = dist*math.tan(thetax) #gets half the distance of the frame
        mapx = objectplanex*2/Resolutionx #gets cm/pixel at that distance, have to multiply in the start to get the full distance of the frame


        FOVy = 32.1*math.pi/180 #FOV in degrees determined experimentall, converts FOV in radians
        thetay = FOVy/2 
        Resolutiony = self.campixelheight #by default 480
        objectplaney = dist*math.tan(thetay)
        mapy = objectplaney*2/Resolutiony #gets cm/pixel at that distance

        for (x,y) in eyecentres:
            eyedist.append([mapx*x, mapy*y])
        try:
            diffx = abs(eyedist[0][0] - eyedist[1][0])
            diffy = abs(eyedist[0][1] - eyedist[1][1])
        except IndexError:
            print("Only 1 eye to clalculate, no difference!")
            diffx = 0
            diffy = 0
        if debug:
            #print(eyecentres)
            #print("Eye Maps:\n", mapx, mapy)
            print("Eye Distances:\n",eyedist)
            print("X difference:\n", diffx)
            print("Y difference:\n", diffy)
        #Mario's way
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
width = 139 #total width of windshield
height = 71.5 #total height of windshield
cam_perp_dist = 60 #perpendicular distance from camera to windshield in cm
cam_spacing = 10
cam_x_dist = width/2 - cam_spacing/2 #horizontal distance from left edge of windshield to camera
#^ 65 cm corresponds to the left camera with a spacing of 10 cm between the two. 
cam_y_dist = height/2 #camera should be centred vertically accoutning for the incline of the windshield
x_centreline = width/2 #half of windshield width
FOV = 2*math.atan((width-cam_x_dist)/cam_perp_dist)*180/math.pi #total field of view of the camera in radians
driver_dist = 20 #horizontal distance from driver to left camera
driver_h = 40 #vertical distance (height) of driver above cameras

#start objects from other Classes
vs = PiCamThreaded().start()
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

#generate lookup table for black bar size using map1 and map2
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





#debug = (Distance, eye array,etc)
#debug (1,0,1,0,1)
#######Head tracking flopped part ends###################################################

#loop until user quits
#increment = 0
while True:
    
    #######################Head tracking copy paste ends#########################

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



        #shift final_x and final_y for angle calculations
        x_shifted = FinalX - row_blk_right[FinalY] - 60 #-60 is the calibration factor for windshield not being exactly in frame
        #air gap of size approx 50 at left edge

        #this one has air gap on bottom AND top, ALSO y is inverted!
        y_shifted = FinalY - col_blk_top[FinalX] - 55  #-55 also calib factor

        x_windshield = x_shifted/(x_res-row_blk_left[FinalY]-row_blk_right[FinalY]-60)*width #these also have the calib factor
        #crop bottom and top air gaps and convert to cm. #55 is top gap, 29 is bottom
        y_windshield = y_shifted/(y_res-col_blk_top[FinalX]-col_blk_bot[FinalX] -55-35)*height
      
        #Convert adjust windshield locations to angle with respect to the camera's normal
        #position on windshield
        #Also convert angles to degrees
        alpha.append(math.atan((x_windshield-cam_x_dist)/cam_perp_dist)*180/math.pi)
        phi.append(-math.atan((y_windshield-cam_y_dist)/cam_perp_dist)*180/math.pi) #has a negative sign to account for inverted FinalY
        
        #print angles for current light
        #print(str(row_blk_left[int(FinalY)]))
        print('alpha ' + str(i+1) + ': ' + str(alpha[i]))
        print('phi: ' + str(i+1) + ': ' + str(phi[i]))
       
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

        Xpoint2 = x_read
        Ypoint2 = y_read    

        #Take difference between rounded map1 and Xpoint
        Xdiff2 = abs(round1-Xpoint2)
                
        #Xmins is 2 arrays corresponding to the X and Y indices of closest match between Xpoint & Map1 
        Xmins2 = np.where(Xdiff2 <= Xdiff2.min() + 1)        
               
        #list of values in map 2 corresponding to Xmins
        Ypos2 = round2[Xmins2[0],Xmins2[1]]
        
        #Take the difference between the list of matched Y positions and Ypoint
        Ydiff2 = abs(Ypos2-Ypoint2)
        
        #Get the index of best match between Ypos and Ypoint
        minidx = np.where(Ydiff2 == Ydiff2.min())
              
        #Final indices for map1/map2
        #These are the final undistorted x,y values of the light source. 
        FinalX2 = Xmins2[1][minidx][0]
        FinalY2 = Xmins2[0][minidx][0] #Y is inverted in terms of a normal x-y plane
        #print FinalX, FinalY

        #shift final_x and final_y for angle calculations
        x_shifted_other = FinalX2 - row_blk_left[FinalY2] #-50 is the calibration factor for windshield not being exactly in frame
        print('x_shifted2: ' + str(x_shifted_other))
        #air gap of size approx 50 at left edge

        #this one has air gap on bottom AND top, AAANDDD y is inverted!
        y_shifted_other = FinalY2 - col_blk_top[FinalX2] - 80  #-55 also calib factor
        print('y_shifted2: ' + str(y_shifted_other))

        x_windshield_other = x_shifted_other/(x_res-row_blk_left[FinalY2]-row_blk_right[FinalY2] - 105)*width #these also have the calib factor
        #crop bottom and top air gaps and convert to cm. #55 is top gap, 29 is bottom
        y_windshield_other = y_shifted_other/(y_res-col_blk_top[FinalX2]-col_blk_bot[FinalX2] -80 - 60)*height
      

        #convert to distance on windshield
        #x_windshield_other = 1.0*x_read/x_res*width
        #y_windshield_other = 1.0*(y_res-y_read)/y_res*height #in the process uninvert y
        #now convert read x,y to angles
        #note addition of cam_spacing to get correct alpha of right camera
        alpha_other = math.atan((x_windshield_other-(cam_x_dist + cam_spacing))/cam_perp_dist)*180/math.pi
        phi_other = -math.atan((y_windshield_other-cam_y_dist)/cam_perp_dist)*180/math.pi
        print('other alpha: ' + str(alpha_other))
        print('other phi: ' + str(phi_other))

    #parallax correction math
    #only works for a single light for now
    #naming conventions from module report
    #actual trig is slightly modified
    
    if len(alpha) == 1 and len(phi) == 1 and alpha_other and phi_other:
        #horizontal part:
        theta_c1 = (90-alpha[0])*math.pi/180 #convert each to radians
        theta_c2 = (90+alpha_other)*math.pi/180
        theta_SC = math.pi-theta_c1-theta_c2
        d_c1 = cam_spacing/math.sin(theta_SC)*math.sin(theta_c2)
        d = d_c1*math.sin(theta_c1) #the key perp distance to light source
        #now get the horizontal driver angle
        d_D = math.sqrt(math.pow(d_c1,2) + math.pow(driver_dist,2)-2*d_c1*driver_dist*math.cos(math.pi-theta_c1))  
        theta_Dx = math.asin(d_c1/d_D*math.sin(math.pi-theta_c1))
    
        #vertical part:
        phi_c = (phi[0] + phi_other)/2.0*math.pi/180 #take the average and convert to radians
        a = d*math.tan(phi_c)
        c = math.sqrt(math.pow(a,2) + math.pow(d,2))
        #now get the vertical driver angle
        s = math.sqrt(math.pow(c,2) + math.pow(driver_h,2) - 2*c*driver_h*math.cos(math.pi/2-phi_c))
        theta_Dy = math.asin(c/s*math.sin(math.pi/2-phi_c))

        #convert to alpha,phi convention and to degrees
        alpha_D = (math.pi/2-theta_Dx)*180/math.pi
        phi_D = (theta_Dy - math.pi/2)*180/math.pi

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
