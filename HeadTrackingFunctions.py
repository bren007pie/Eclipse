#sourced from: https://www.instructables.com/id/Face-and-Eye-Detection-With-Raspberry-Pi-Zero-and-/

#if getting an error run:
#sudo modprobe bcm2835-v4l2
#before running the script



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
##        try:
##            watchdog = Watchdog(5) #watchdog timer waits 2 seconds
##            distance()
##            print("Ultrasonic Sensor is hooked up!")
##        except:
##            print("Waited too long, No Ultrasonic sensor detected")
##            Uenable = False #disables all ultrasonic stuff
##        watchdog.stop()
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

