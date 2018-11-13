import RPi.GPIO as GPIO
import time
import numpy as np

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(0)

#list of pins to be used as outputs
channels = list(range(20,27))
##channels = np.arange(20,27)
GPIO.setup(channels, GPIO.OUT, initial = 0)

#list of pins corresponding to row and column voltage control
rowVoltagePins = channels[:3]
columnVoltagePins = channels[3:]

#set frequency of oscillation
freq = 50
halfT = 0.5/freq

#input desired opaque coordinates
x1,y1 = 3,0
x2,y2 = 0,2

try:
    while 1:
        
        if (x1 != x2) and (y1 != y2):
            GPIO.output(rowVoltagePins[y1],1)
            GPIO.output(columnVoltagePins[x1],0)
            time.sleep(halfT)
        
            GPIO.output(rowVoltagePins[y1],0)
            GPIO.output(columnVoltagePins[x1],1)
            time.sleep(halfT)
            
        elif x1 == x2:
            print("ya facked it, mate")
            
        elif y1 == y2:
            print("BadBadNotGood")
        
except KeyboardInterrupt:
    GPIO.cleanup()
