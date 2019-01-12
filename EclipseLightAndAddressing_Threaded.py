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

#TODO
#CONVERT ANGLES TO X-Y COORDS IN CENTIMETERS
#BIN ANGLES/COORDS TO GRID TILES
#UPDATE SERIAL COMMUNICATION TO ACCEPT DRIVER LOCATION


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


########################################


class GridAddressing:
    
    def __init__(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(0)

        #list of pins to be used as outputs
        channels = list(range(4,25))
        GPIO.setup(channels, GPIO.OUT, initial = 1)

        #list of pins corresponding to row and column voltage control
        self.rowVoltagePins = channels[:7]
        self.columnVoltagePins = channels[7:]

        #set frequency of oscillation
        freq = 1
        self.halfT = 0.49/freq


    def start(self):
        t = Thread(target = self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        while True:
            x1,y1 = 0,0
            x2,y2 = 1,1
            x3,y3 = None,None
            x4,y4 = None,None
            
            if x2 == None:
                print("only one location")
                self.flicker2([self.columnVoltagePins[x1]],[self.rowVoltagePins[y1]])
        
        
            elif x3 == None:
                if (abs(x1-x2) == 1) and (abs(y1-y2) == 1):
                    print("four-tile square")
                    self.flicker4([self.columnVoltagePins[x1],self.columnVoltagePins[x2]], [self.rowVoltagePins[y1],self.rowVoltagePins[y2]], True)
                
                elif (x1 != x2) and (y1 != y2):
                    print("two locations uncommon")
                    fself.flicker4([self.columnVoltagePins[x1],self.columnVoltagePins[x2]], [self.rowVoltagePins[y1],self.rowVoltagePins[y2]], False)
                
                elif x1 == x2:
                    print("x1 = x2")
                    self.flicker3([self.columnVoltagePins[x1]],[self.rowVoltagePins[y1],self.rowVoltagePins[y2]])
                
                elif y1 == y2:
                    print("y1 = y2")
                    self.flicker3([self.rowVoltagePins[y1]],[self.columnVoltagePins[x1],self.columnVoltagePins[x2]])
                
                
            #potentially incomplete: might require redundancy of which location makes up the four-tile square
            elif x4 == None:
                print("four-tile square + one extra location")
                if (abs(x1-x2) == 1) and (abs(y1-y2) == 1):
                    self.flicker6([self.columnVoltagePins[x1],self.columnVoltagePins[x2],self.columnVoltagePins[x3]], [self.rowVoltagePins[y1],self.rowVoltagePins[y2],self.rowVoltagePins[y3]])
                 
                 
            #potentially incomplete: might require redundancy of which locations make up each four-tile square
            else:
                print("2 four-tile squares")
                if (abs(x1-x3) >= 2 and abs(x2-x4)) and (abs(y1-y3) >= 2 and abs(y2-y4)):
                    self.flicker8([self.columnVoltagePins[x1],self.columnVoltagePins[x2],self.columnVoltagePins[x3],self.columnVoltagePins[x4]], [self.rowVoltagePins[y1],self.rowVoltagePins[y2],self.rowVoltagePins[y3],self.rowVoltagePins[y4]], False)
                
                elif ((x1 == x3) and (x2 == x4)) != ((y1 == y3) and (y2 == y4)):
                    self.flicker8([self.columnVoltagePins[x1],self.columnVoltagePins[x2],self.columnVoltagePins[x3],self.columnVoltagePins[x4]], [self.rowVoltagePins[y1],self.rowVoltagePins[y2],self.rowVoltagePins[y3],self.rowVoltagePins[y4]], True)    

        return

    def flicker2(self, X, Y):
        GPIO.output([X[0],Y[0]],[0,1])
        time.sleep(self.halfT)
        GPIO.output([X[0],Y[0]],[1,0])
        time.sleep(self.halfT)

    def flicker3(self, X, Y):
        GPIO.output([X[0],Y[0],Y[1]],[0,1,1])
        time.sleep(self.halfT)
        GPIO.output([X[0],Y[0],Y[1]],[1,0,0])
        time.sleep(self.halfT)

    def flicker4(self, X, Y, inPhase):
        if inPhase == True:
            GPIO.output([X[0],Y[0],X[1],Y[1]],[0,1,0,1])
            time.sleep(self.halfT)
            GPIO.output([X[0],Y[0],X[1],Y[1]],[1,0,1,0])
            time.sleep(self.halfT)
           
        elif inPhase == False:
            GPIO.output([X[0],Y[0],X[1],Y[1]],[0,1,1,0])
            time.sleep(self.alfT)
            GPIO.output([X[0],Y[0],X[1],Y[1]],[1,0,0,1])
            time.sleep(self.halfT)
            
    def flicker6(self, X, Y):
        GPIO.output([X[0],Y[0],X[1],Y[1],X[2],Y[2]],[0,1,0,1,1,0])
        time.sleep(self.halfT)
        GPIO.output([X[0],Y[0],X[1],Y[1],X[2],Y[2]],[1,0,1,0,0,1])
        time.sleep(self.halfT)
        
    def flicker8(self, X, Y, inPhase):
        if inPhase == True:
            GPIO.output([X[0],Y[0],X[1],Y[1],X[2],Y[2],X[3],Y[3]],[0,1,0,1,0,1,0,1])
            time.sleep(self.halfT)
            GPIO.output([X[0],Y[0],X[1],Y[1],X[2],Y[2],X[3],Y[3]],[1,0,1,0,1,0,1,0])
            time.sleep(self.halfT)
           
        elif inPhase == False:
            GPIO.output([X[0],Y[0],X[1],Y[1],X[2],Y[2],X[3],Y[3]],[0,1,0,1,1,0,1,0])
            time.sleep(self.halfT)
            GPIO.output([X[0],Y[0],X[1],Y[1],X[2],Y[2],X[3],Y[3]],[1,0,1,0,0,1,0,1])
            time.sleep(self.halfT)
                  
    def stop(self):
        self.stopped = True
        GPIO.cleanup()

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

#This function takes in raw pixel coordinates and undistorts them
def undistort(Xpoint, YPoint):
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
    
#This function accounts for black bars from undistortion.
######SOME PARTS ARE FOR AIR GAPS WHICH SHOULDN'T EXIST ANY MORE)
#It returns the corrected x,y values in terms of cm on the windshield
def shift(FinalX, FinalY):
    #shift final_x and final_y for angle calculations
    x_shifted = FinalX - row_blk_right[FinalY] - 60 #-60 is the calibration factor for windshield not being exactly in frame
    #air gap of size approx 50 at left edge

    #this one has air gap on bottom AND top, ALSO y is inverted!
    y_shifted = FinalY - col_blk_top[FinalX] - 55  #-55 also calib factor

    x_windshield = x_shifted/(x_res-row_blk_left[FinalY]-row_blk_right[FinalY]-60)*width #these also have the calib factor
    #crop bottom and top air gaps and convert to cm. #55 is top gap, 29 is bottom
    y_windshield = y_shifted/(y_res-col_blk_top[FinalX]-col_blk_bot[FinalX] -55-35)*height
    return(x_windshield,y_windshield)
    

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
ga = GridAddressing().start()
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
#increment = 0

flag = 3 #default serial flag before loop starts
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

        (FinalX, FinalY) = undistort(cX,cY)               
        (x_windshield,y_windshield) = shift(FinalX,FinalY)
              
        #Convert adjusted windshield locations to angle with respect to the camera's normal
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
    
    #reset previously read list of alphas,phis
    #before reading from the serial buffer, check if the last flag was 3. If so, that
    #means a fresh new batch of info is coming so need to clear the below arrays.
    #The next time the buffer is read it should be the 1 flag.

    if flag == 3:      
        alpha_other = [None,None]
        phi_other = [None,None]
        x1_array = []
        x2_array = []
        y1_array = []
        y2_array = []
        dx_array = []
        dy_array = []
        dz_array = []
    #perform serial read operations if something is in the buffer
    if ser.inWaiting():

        #save integers 1,2,3 as flag if they come in
        #if prev integer was 1, it's light 1
        #if prev integer was 2, it's light 2
        #if prev integer was 3, it's driver
        #if no flag is detected then continue without further reading the serial buffer
        #reset key arrays after math is done if the last flag was 3

        flag = ser.read() #now read from the serial buffer to decide what to do next

        if  flag == 1:
            current_read = ser.read() #save current read value so it's not lost somehow after checking it
            if current_read != 1: #good chance that don't actually need to wait for this to happen
                x1_array.append(current_read) #add first byte
                x1_array.append(ser.read()) #add second byte
                for i in range(2):
                    y1_array.append(ser.read())

                x1_packed = x1_array[0] + x1_array[1]
                x1_unpacked = struct.unpack('h', x1_packed)
                x1_read = x1_unpacked[0]
              
                y1_packed = y1_array[0] + y1_array[1]
                y1_unpacked = struct.unpack('h', y1_packed)
                y1_read = y1_unpacked[0]

                (FinalX_other1,FinalY_other1) = undistort(x1_read,y1_read)
                (x_windshield_other1, y_windshield_other1) = shift(FinalX_other1,FinalY_other1)

                #now convert read x,y to angles
                #note addition of cam_spacing to get correct alpha of right camera
                alpha_other[0] = math.atan((x_windshield_other1-(cam_x_dist + cam_spacing))/cam_perp_dist)*180/math.pi
                phi_other[0] = -math.atan((y_windshield_other1-cam_y_dist)/cam_perp_dist)*180/math.pi
                #print('other alpha 1: ' + str(alpha_other[0]))
                #print('other phi 1: ' + str(phi_other[0]))
                
           
        if flag == 2:
            current_read = ser.read()
            if current_read != 2:
                x2_array.append(current_read)
                x2_array.append(ser.read())
                for i in range(2):
                    y2_array.append(ser.read())

                x2_packed = x2_array[0] + x2_array[1]
                x2_unpacked = struct.unpack('h', x2_packed)
                x2_read = x2_unpacked[0]

                y2_packed = y2_array[0] + y2_array[1]
                y2_unpacked = struct.unpack('h', y2_packed)
                y2_read = y2_unpacked[0]

                (FinalX_other2,FinalY_other2) = undistort(x2_read,y2_read)
                (x_windshield_other2, y_windshield_other2) = shift(FinalX_other2,FinalY_other2)
                
                #now convert read x,y to angles
                #note addition of cam_spacing to get correct alpha of right camera
                alpha_other[1] = math.atan((x_windshield_other2-(cam_x_dist + cam_spacing))/cam_perp_dist)*180/math.pi
                phi_other[1] = -math.atan((y_windshield_other2-cam_y_dist)/cam_perp_dist)*180/math.pi
                #print('other alpha 2: ' + str(alpha_other[1]))
                #print('other phi 2: ' + str(phi_other[1]))
                    
        if flag == 3:          
            current_read = ser.read()
            if current_read != 3:
                dx_array.append(current_read)
                dx_array.append(ser.read())
                for i in range(2):
                    dy_array.append(ser.read())
                for i in range(2):
                    dz_array.append(ser.read())
        
                dx_packed = dx_array[0] + dx_array[1]
                dx_unpacked = struct.unpack('h', dx_packed)
                dx_read = dx_unpacked[0]

                dy_packed = dy_array[0] + dy_array[1]
                dy_unpacked = struct.unpack('h', dy_packed)
                dy_read = dy_unpacked[0]

                dz_packed = dz_array[0] + dz_array[1]
                dz_unpacked = struct.unpack('h', dz_packed)
                dz_read = dz_unpacked[0]/10.0 #divide by 10 since it was multiplied by 10 before being sent. 
        
                     
    ########HUGE OVERHAUL FOR PARALLAX IS NEEDED#################### (Useless as it is now)
    #########SAVE FOR AFTER INTEGRATION???#########################

    #parallax correction math
    #ONLY WORKS FOR A SINGLE LIGHT FOR NOW
    #naming conventions from module report
    #actual trig is slightly modified
    
##    if len(alpha) == 1 and len(phi) == 1 and alpha_other[0] and phi_other[0]:
##        #horizontal part:
##        theta_c1 = (90-alpha[0])*math.pi/180 #convert each to radians
##        theta_c2 = (90+alpha_other[0])*math.pi/180
##        theta_SC = math.pi-theta_c1-theta_c2
##        d_c1 = cam_spacing/math.sin(theta_SC)*math.sin(theta_c2)
##        d = d_c1*math.sin(theta_c1) #the key perp distance to light source
##        
##        #now get the horizontal driver angle
##        d_D = math.sqrt(math.pow(d_c1,2) + math.pow(driver_dist,2)-2*d_c1*driver_dist*math.cos(math.pi-theta_c1))  
##        theta_Dx = math.asin(d_c1/d_D*math.sin(math.pi-theta_c1))
##    
##        #vertical part:
##        phi_c = (phi[0] + phi_other[0])/2.0*math.pi/180 #take the average and convert to radians
##        a = d*math.tan(phi_c)
##        c = math.sqrt(math.pow(a,2) + math.pow(d,2))
##        
##        #now get the vertical driver angle
##        s = math.sqrt(math.pow(c,2) + math.pow(driver_h,2) - 2*c*driver_h*math.cos(math.pi/2-phi_c))
##        theta_Dy = math.asin(c/s*math.sin(math.pi/2-phi_c))
##
##        #convert to alpha,phi convention and to degrees
##        alpha_D = (math.pi/2-theta_Dx)*180/math.pi
##        phi_D = (theta_Dy - math.pi/2)*180/math.pi
##
##        #print('Driver alpha: ' + str(alpha_D))
##        #print('Driver phi: ' + str(phi_D))

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
ga.stop()
