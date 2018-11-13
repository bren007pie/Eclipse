#sourced from: https://www.instructables.com/id/Face-and-Eye-Detection-With-Raspberry-Pi-Zero-and-/

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

#My Global variables
symthresh = 5 #pixel distance away from average of symetry threshold for eye filtering
framebuffer = []
framebufferlen = 5 #maybe can just make this bigger and not take the average but the most frequent value? Don't want averages to be squed by far values
areathreshold = 5 #radius in pixels of the acceptable into accept

#Objects
class Watchdog: #watchdog timer to stop the program if ultrasonics are not hooked up
    def __init__(self, timeout, userHandler=None):  # timeout in seconds
        self.timeout = timeout
        self.handler = userHandler if userHandler is not None else self.defaultHandler
        self.timer = Timer(self.timeout, self.handler)
        self.timer.start()

    def reset(self):
        self.timer.cancel()
        self.timer = Timer(self.timeout, self.handler)
        self.timer.start()

    def stop(self):
        self.timer.cancel()

    def defaultHandler(self):
        raise self

#Setup Functions

def ultrasonicsetup(enable): #setups up ultrasonic range finder, if enable is false it doesn't run
    global Uenable, GPIO_TRIGGER, GPIO_ECHO  #setups a global variables. Uenable that stops all ultrasonic stuff
    Uenable = enable
    if enable:
        GPIO.setwarnings(False) #This disables the warning about channel being in use
        GPIO.setmode(GPIO.BCM) #GPIO Mode (BOARD or BCM)
        #set GPIO Pins
        GPIO_TRIGGER = 18
        GPIO_ECHO = 24
        #set GPIO direction (IN / OUT)
        GPIO.setup(GPIO_TRIGGER, GPIO.OUT)
        GPIO.setup(GPIO_ECHO, GPIO.IN)
        try:
            watchdog = Watchdog(2) #watchdog timer waits 2 seconds
            distance()
            print("Ultrasonic Sensor is hooked up!")
        except:
            print("Waited too long, No Ultrasonic sensor detected")
            Uenable = False #disables all ultrasonic stuff
        watchdog.stop()
    return

def distance(): #gets distance from the ultrasonic range finder
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
    centres = [float(faces[0][0] + faces[0][2]/2), float(faces[0][1]+ faces[0][3]/2)] #gets x and y of face centre
    if debug:
        print("Face array:\n", faces) #prints the contents of the faces arrays
        #print("Number faces:\n", numfaces) #not a thing anymore
        print("Face centre:\n", centres)
    return centres

def geteyecentres(eyes, debug):
    centres = []

    return centres


def geteyedistance(eyecentres, dist, debug):
    #FOV 60 degrees (assume both directions but test) #https://support.logitech.com/en_us/article/17556
    #example eyecentres
    #eyecentres = [[100, 300], [200,300]] #left and right eye
    #dist = 20 #person is 20 cm away
    #resolution = 1280 in x 720 in y
    #FOV 60 degrees (assume both directions but test) #https://support.logitech.com/en_us/article/17556

    #My way
    eyedist = []
    FOVx = 60*m.pi/180 #FOV in radians
    thetax = FOVx/2
    Resolutionx = 1280

    objectplanex = dist*m.tan(thetax)
    mapx = objectplanex/Resolutionx #gets cm/pixel at that distance


    FOVy = 60*m.pi/180 #FOV in radians
    thetay = FOVx/2
    Resolutiony = 720

    objectplaney = dist*m.tan(thetay)
    mapy = objectplaney/Resolutiony #gets cm/pixel at that distance

    for (x,y) in eyecentres:
        eyedist.append([mapx*x, mapy*y])
    if debug:
        #print(eyecentres)
        print("Eye Maps:\n", mapx, mapy)
        print("Eye Distances:\n",eyedist)
    return eyedist


    


#program start


ultrasonicsetup(False) #True if ultrasonic is hooked up, false if otherwise
fps = FPS().start() #defines the FPS object
testIRled()



#main loop

while 1:
    ret, img = cap.read() #reads in the image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #converts to grayscale
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) #detects faces in the grayscale image

    try: #error handeling for no faces in the frame
        
        if Uenable: #if ultrasonics are on
            dist = distance() #does the ultrasonic distance measurement
        elif Uenable == False:
            dist = -1 #-1 used to indicate ultrasonic sensor is not on or can fill with a set value in cm can still work

        facecentre = getfacecentre(faces,True) #debug prints out face info
        
        

        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),1) #draws the rectangle around the corners in blue to the image object
            cv2.circle(img, (int(x+w/2),int(y+h/2)), 3, (255,0,0),2) #draws a blue circle of radius 3 in the centre of the face to the image object. Pixel reference need to be an integer
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]

            eyes = eye_cascade.detectMultiScale(roi_gray) #detects the eyes

            numeyes = int(eyes.size/4) #finds the number of eyes, is a numpy.ndarray, getting the number of ellements and dividing by 4
           

            #eyecentres = geteyecentres(eyes,Debug)
            #Getting the Eye Centres makes an array (should just make this a function)
            eyecentres = [] #empty list, defined each time
            for (ex,ey,ew,eh) in eyes:
                eyecentres.append([x+ex+ew/2, y+ey+eh/2])
            print("EyeCentres:\n", eyecentres)

            eyedistance = geteyedistance(eyecentres, 20*2.54, True)

            #finding the number of eyes, should also just be a function
            print("Eye array: \n", eyes)
            print(numeyes, " eyes detected")

            #print("Framebuff:\n", framebuffer )
            #gets the average of the last frames buffer before it's rolled over/updated. Should be a function. Consider not using an average? Idk what method would be better. Maybe threshold via variance?
            if len(framebuffer) == framebufferlen: #checks to make sure the frame buffer is full
                avgeyex = 0 #the average x location of the eye, reinitalized to 0
                avgeyey = 0 #the average y location of the eye, reinitializes to 0
                bufeyenum = 0
                for i in range(framebufferlen): #itterates through the frame buffer to get average. range(x) goes through 0,1,2..,(x-1)
                    for j in range(len(framebuffer[i])):
                        avgeyex = avgeyex + float(framebuffer[i][j][0]) #sums up all the eyecentre xs, is a triple nested list, remember index from 0, can't acess these until buffer has at least 1 element!
                        avgeyey = avgeyey + float(framebuffer[i][j][1]) #sums up all the eyecentre ys,  is a triple nested list, remember index from 0, can't acess these until buffer has at least 1 element!
                        bufeyenum = bufeyenum + 1
                avgeyex = avgeyex/bufeyenum
                avgeyey = avgeyey/bufeyenum
                print("Average x location: \n", avgeyex)
                print("Average y location: \n", avgeyey)


            #updates a rolling framebufferlen long buffer of detected eyes to check check
            framebuffer.append(eyecentres)
            if len(framebuffer) > framebufferlen: #lengh of the frame buffer detected by the top most arrays
                framebuffer.pop(0) #pops off the left most index to make a rolling buffer


            #trace through this first!
            #Filtering out eyes below the centre of the face, should be a function. why doesn't this do this already?
            for i in range(numeyes):
                if eyecentres[i][1] > facecentre[1]: #if the eye is below the face centre (greater than because negatively indexed)
                    #print(eyecentres[i][1]) prints the value it deletes
                    cv2.circle(img, (int(eyecentres[i][0]),int(eyecentres[i][1]) ), 2, (0,0,0),3) #draws the eye to be deleted in black
                    eyes = np.delete(eyes,i,0) #deletes the eye it detects is bad
                    i = i-1 #this reshifts the index so you don't run into indexing errors after deleting
                    print("Eye", i+1 , "below nose! Deleted!")
                    numeyes = int(eyes.size/4) #updates number of eyes

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


            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),1)
                cv2.circle(img, (int(x+ex+ew/2),int(y+ey+eh/2)), 2, (0,255,0),1) #draws a green circle of radius 2 in the centre of the eye to the image object. Pixel reference need to be an integer.
                #all eye dimensions are with reference to the face so need to add the face coord to each one
                ###print("Eye Position:",(x+ex+ew/2, y+ey+eh/2,"%.1f cm" % dist)) #prints the middle of the eye location


        cv2.imshow('img',img) #img reference is 0,0 in top left
        k = cv2.waitKey(30) & 0xff
        if k == 27: #press escape key to escape
            break
 

        fps.update() #updates the fps object each time the detection runs

    except: #except if it throws the face error and print that
        cv2.imshow('img',img) #img reference is 0,0 in top left
        k = cv2.waitKey(30) & 0xff
        if k == 27: #press escape key to escape
            break




        fps.update() #updates the fps object each time the detection runs

fps.stop() #stops the fps counting
print("[INFO] approx FPS: {:.2f}".format(fps.fps()))
#cleans up
cap.release()
cv2.destroyAllWindows()

#if getting an error run:
#sudo modprobe bcm2835-v4l2
#before running the script

#Python lists
#https://docs.python.org/2/tutorial/datastructures.html
#NumPy Array Functions
#https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.array-manipulation.html


#Webcam take a picture: fswebcam -r 1280x720 --no-banner image3.jpg
#Video location /dev/video0 , location of the picam, /dev/video1 is the webcam
