import math
import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(0)

#list of pins to be used as outputs
channels = list(range(4,18))
GPIO.setup(channels, GPIO.OUT, initial = 0)

#list of pins corresponding to row and column voltage control
bottomRowPins = channels[:7]
topRowPins = channels[7:]

#constants and variables
xCam1 = 50
yCam2 = 50
dSensorToWindshield = 20
acceptableError = 3

#outputs from Light Tracking (in deg)
alpha1 = -40
phi1 = -15
alpha2 = None
phi2 = None

#outputs from Driver Tracking (in cm)
horizontalDisplacement = 0
verticalDisplacement = 0
depthSensorToDriver = 60

#loop
while 1:
    x1 = y1 = x2 = y2 = x3 = y3 = x4 = y4 = None
    
    dDriver = dSensorToWindshield + depthSensorToDriver
    xDriver = xCam1 + horizontalDisplacement
    yDriver = yCam2 + verticalDisplacement
    
    if type(alpha1) != type(None):
        xProjection1 = dDriver*math.tan(math.radians(alpha1)) + xDriver
        yProjection1 = dDriver*math.tan(math.radians(phi1)) + yDriver
        
        column1 = int(xProjection1/20)
        row1 = int(yProjection1/20)
        
        if (column1 >= 0 and column1 < 7) and (row1 == 1 or row1 == 2):
            print("blah1")
            x1 = column1
            y1 = row1
        
        if type(alpha2) != type(None):
            xProjection2 = dDriver*math.tan(math.radians(alpha2)) + xDriver
            yProjection2 = dDriver*math.tan(math.radians(phi2)) + yDriver
            
            column2 = int(xProjection2/20)
            row2 = int(yProjection2/20)
            
            if (column2 >= 0 and column2 < 7) and (row2 == 1 or row2 == 2):
                print("blah2")
                x2 = column2
                y2 = row2
             
    #begin actual grid addressing
    if type(x1) != type(None):
        if y1 == 1:
            GPIO.output(bottomRowPins[x1],1)
        elif y1 == 2:
            GPIO.output(topRowPins[x1],1)
            
        if type(x2) != type(None):
            if y1 == 1:
                GPIO.output(bottomRowPins[x2],1)
            elif y1 == 2:
                GPIO.output(topRowPins[x2],1)
        
    else:
        GPIO.output(bottomRowPins,0)
        GPIO.output(topRowPins,0)
        
