#Purpose: This is the main code for head tracking for the Eclipse Blocking System.
#Author: Brendan Fallon 
#Date: Dec 2018


#if getting an error run:
#sudo modprobe bcm2835-v4l2
#before running the script

#Imports and libraries
import numpy as np
import cv2
import os
from imutils.video import FPS
#Ultrasonic imports
import RPi.GPIO as GPIO
import time
from threading import Timer
import math as m
#doesn't make much sense to have a headtracking functions file as things are mostly setup as global variables and such

#global variable setup
#Sets up Camera and OpenCV and FPS
os.chdir("/usr/local/share/OpenCV/haarcascades/")
face_cascade = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_eye.xml')
cap = cv2.VideoCapture("/dev/video0") #sets up video capture, video1 is the webcam, video0 is the picam
    #camera has atributes set by cap.set(cv2.CAP_PROP_EXPOSURE, 40) #Reference: https://docs.opencv.org/trunk/d4/d15/group__videoio__flags__base.html#gaeb8dd9c89c10a5c63c139bf7c4f5704d
    #CAP_PROP_BRIGHTNESS
    #CAP_PROP_CONTRAST
    #CAP_PROP_SATURATION
    #CAP_PROP_GAIN
    #CAP_PROP_AUTO_EXPOSURE ?
    #no exposure, no monochrome, iso speed
    #Sets the camera height to 720p
    ##cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    ##cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

campixelwidth = cap.get(3) #cv2.CAP_PROP_FRAME_WIDTH
print("Pixel width:\n", campixelwidth)
campixelheight = cap.get(4)  #cv2.CAP_PROP_FRAME_HEIGHT
print("Pixel height:\n", campixelheight)

#My Global variables
symthresh = 5 #pixel distance away from average of symetry threshold for eye filtering
framebuffer = []
framebufferlen = 5 #maybe can just make this bigger and not take the average but the most frequent value? Don't want averages to be squed by far values
areathreshold = 5 #radius in pixels of the acceptable into accept

#Objects
#watchdog timer removed 

#Ultrasonic Functions - need to be in code for 

def ultrasonicsetup(enable,manualdistance): #setups up ultrasonic range finder, if enable is false it doesn't run
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

def distance(debug): #gets distance from the ultrasonic range finder
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


def testIRled(): ##Test IR output
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(0)
    channels = [26] #26 is the last one
    GPIO.setup(channels, GPIO.OUT, initial = 1)
    GPIO.output(channels, True)
    return

#Program Functions #ALWAYS NAME FUNCTIONS DIFFERENT THAN VARIABLE NAMES!
        

#makes it crash for some reason      
def waitfor1face(faces): #limiting to 1 face makes it infinite loop for some reason, 
    numfaces = int(faces.size/4)
    while numfaces > 1: #stops the program until there is only 1 face in the frame
        print("Please try to make there only be 1 face")
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        numfaces = int(faces.size/4)
    return

def getfacecentre(faces,debug): #gets centre of the face and returns it as an x and y list
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

def geteyecentres(eyes, debug): #Getting the Eye Centres makes an array (should just make this a function)
    
    centres = [] #empty list, defined each time
    for (ex,ey,ew,eh) in eyes:
        centres.append([x+ex+ew/2, y+ey+eh/2])
    if debug:
        print("EyeCentres:\n", centres)

    return centres

def getnumeyes(eyes,debug):
    try: #error checking if there is no eyes detected
        num = int(eyes.size/4) #finds the number of eyes, is a numpy.ndarray, getting the number of ellements and dividing by 4
        if debug:
            print(num, " eyes detected")
    except AttributeError:
        num = -1
        print("no faces detected! Set to -1!")
    return num

def geteyedistance(eyecentres, dist, debug):
    #FOV 60 degrees (assume both directions but test) #https://support.logitech.com/en_us/article/17556
    #example eyecentres
    #eyecentres = [[100, 300], [200,300]] #left and right eye
    #dist = 20 #person is 20 cm away
    #resolution = 1280 in x 720 in y
    #FOV 60 degrees (assume both directions but test) #https://support.logitech.com/en_us/article/17556

    #My way
    eyedist = []
    FOVx = 42.6*m.pi/180 #FOV in degrees determined experimentall, converts FOV in radians
    thetax = FOVx/2
    Resolutionx = campixelwidth #by default 640
    objectplanex = dist*m.tan(thetax) #gets half the distance of the frame
    mapx = objectplanex*2/Resolutionx #gets cm/pixel at that distance, have to multiply in the start to get the full distance of the frame


    FOVy = 32.1*m.pi/180 #FOV in degrees determined experimentall, converts FOV in radians
    thetay = FOVy/2 
    Resolutiony = campixelheight #by default 480
    objectplaney = dist*m.tan(thetay)
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

def deleteeyesbelowcentre(numeyes,eyecentres,facecentre,eyes):
    #Filtering out eyes below the centre of the face, should be a function. why doesn't this do this already?
    j = 0 #accumulator for how many eyes deleted, has to index shift for deleted one
    for i in range(numeyes):
        if eyecentres[i][1] > facecentre[1]: #if the eye is below the face centre (greater than because negatively indexed)
            #print("Eye", i , "below nose! Going to be Deleted!")
            #print("Eyes",eyes)
            #print(eyecentres[i][1]) prints the value it deletes
            cv2.circle(img, (int(eyecentres[i][0]),int(eyecentres[i][1]) ), 2, (0,0,0),3) #draws the eye to be deleted in black
            eyes = np.delete(eyes,(i-j),0) #deletes the eye it detects is bad
            j = j + 1 
            print("Eye", i , "below nose! Deleted!", "j =",j)
    return eyes


#program start


ultrasonicsetup(False,0) #True if ultrasonic is hooked up, false if otherwise
fps = FPS().start() #defines the FPS object
testIRled()
#debug = (Distance, eye array,etc)
#debug (1,0,1,0,1)



#main loop

while 1:
    ret, img = cap.read() #reads in the image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #converts to grayscale
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) #detects faces in the grayscale image


    print("I START")
    dist = distance(False) #does the ultrasonic distance measurement, True is the debug
    
    
    

    for (x,y,w,h) in faces: #if there are no faces this doesn't run
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),1) #draws the rectangle around the corners in blue to the image object
        cv2.circle(img, (int(x+w/2),int(y+h/2)), 3, (255,0,0),2) #draws a blue circle of radius 3 in the centre of the face to the image object. Pixel reference need to be an integer
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        facecentre = getfacecentre(faces,False) #debug prints out face info

        eyes = eye_cascade.detectMultiScale(roi_gray) #detects the eyes

        numeyes = getnumeyes(eyes,False) 

        


##        
##
##        #print("Framebuff:\n", framebuffer )
##        #gets the average of the last frames buffer before it's rolled over/updated. Should be a function. Consider not using an average? Idk what method would be better. Maybe threshold via variance?
##        if len(framebuffer) == framebufferlen: #checks to make sure the frame buffer is full
##            avgeyex = 0 #the average x location of the eye, reinitalized to 0
##            avgeyey = 0 #the average y location of the eye, reinitializes to 0
##            bufeyenum = 0
##            for i in range(framebufferlen): #itterates through the frame buffer to get average. range(x) goes through 0,1,2..,(x-1)
##                for j in range(len(framebuffer[i])):
##                    avgeyex = avgeyex + float(framebuffer[i][j][0]) #sums up all the eyecentre xs, is a triple nested list, remember index from 0, can't acess these until buffer has at least 1 element!
##                    avgeyey = avgeyey + float(framebuffer[i][j][1]) #sums up all the eyecentre ys,  is a triple nested list, remember index from 0, can't acess these until buffer has at least 1 element!
##                    bufeyenum = bufeyenum + 1
##            avgeyex = avgeyex/bufeyenum
##            avgeyey = avgeyey/bufeyenum
##            print("Average x location: \n", avgeyex)
##            print("Average y location: \n", avgeyey)
##
##
##        #updates a rolling framebufferlen long buffer of detected eyes to check check
##        framebuffer.append(eyecentres)
##        if len(framebuffer) > framebufferlen: #lengh of the frame buffer detected by the top most arrays
##            framebuffer.pop(0) #pops off the left most index to make a rolling buffer

        eyecentres = geteyecentres(eyes,False)

        eyes = deleteeyesbelowcentre(numeyes,eyecentres,facecentre,eyes) #overrites eyes with the deleted ones


        #needs to update the number of eyes after they are deleted
        numeyes = getnumeyes(eyes,False)
        
##        #looks for symetric eyes and filters out eranous ones. Takes distance, then average, then deletes one not within 5 of average
##        eyedevs = []
##        eyeavg = 0
##        for i in range(numeyes):
##            eyedev = float(abs(eyecentres[i][0] - facecentre[0]))
##            print("Eye Deviation:\n", eyedev)
##            eyedevs.append(eyedev)
##            eveavg = eyeavg + float(eyedev)
##            print("Eye average:\n", eyeavg, type(eyeavg), type(eyedev))
##        eyeavg = sum(eyedevs/numeyes
##        print("Eye D6e9viations:\n", eyedevs)
##        print("Eye average deviation:\n", eyeavg)
##        for i in range(numeyes):
##            if abs(eyedevs[i] - eyeavg) > symthresh:
##                cv2.circle(img, (int(eyecentres[i][0]),int(eyecentres[i][1]) ), 2, (0,0,255),3) #draws the eye to be deleted in red
##                eyes = np.delete(eyes,i,0) #deletes the eye it detects is bad
##                i = i-1 #this reshifts the index so you don't run into indexing errors after deleting
##                print("Eye", i+1 , "not symetric about face! Deleted!")
##                numeyes = int(eyes.size/4) #updates number of eyes
##
##
##
##        print("Eye Deviations:\n",eyedevs)


        #facecentre[0] x position

        #cuts the number of eyes down to 2 "randomly"
        while numeyes > 2:
            eyes = np.delete(eyes,2,0) #deletes the eye at index 2, or the 3rd eye
            numeyes = int(eyes.size/4)

        eyecentres = geteyecentres(eyes,False)


        eyedistance = geteyedistance(eyecentres, dist, True)
        
        for (ex,ey,ew,eh) in eyes: #writes all the circtles
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),1)
            cv2.circle(img, (int(x+ex+ew/2),int(y+ey+eh/2)), 2, (0,255,0),1) #draws a green circle of radius 2 in the centre of the eye to the image object. Pixel reference need to be an integer.
            #all eye dimensions are with reference to the face so need to add the face coord to each one
            ###print("Eye Position:",(x+ex+ew/2, y+ey+eh/2,"%.1f cm" % dist)) #prints the middle of the eye location


    cv2.imshow('img',img) #img reference is 0,0 in top left
    k = cv2.waitKey(30) & 0xff
    if k == 27: #press escape key to escape
        break
    try:
        print("Eye Locations relative to the camera is", ((eyecentres[0][0],eyecentres[0][1],dist),(eyecentres[1][0],eyecentres[1][1],dist) ))
    except:
        print("No eyes in frame!")
    ("I FINISH")
    fps.update() #updates the fps object each time the detection runs





    fps.update() #updates the fps object each time the detection runs

fps.stop() #stops the fps counting
print("[INFO] approx FPS: {:.2f}".format(fps.fps()))
#cleans up
cap.release()
cv2.destroyAllWindows()

###Notes
#sourced from: https://www.instructables.com/id/Face-and-Eye-Detection-With-Raspberry-Pi-Zero-and-/

#if getting an error run:
#sudo modprobe bcm2835-v4l2
#before running the script

#Python lists
#https://docs.python.org/2/tutorial/datastructures.html
#NumPy Array Functions
#https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.array-manipulation.html


#Webcam take a picture: fswebcam -r 1280x720 --no-banner image3.jpg
#Video location /dev/video0 , location of the picam, /dev/video1 is the webcam
