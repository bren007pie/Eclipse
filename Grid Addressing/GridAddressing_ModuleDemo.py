import RPi.GPIO as GPIO
import time
import numpy as np

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(0)

#list of pins to be used as outputs
channels = list(range(20,27))
##channels = np.arange(20,27)
GPIO.setup(channels, GPIO.OUT, initial = 1)

#list of pins corresponding to row and column voltage control
rowVoltagePins = channels[:3]
columnVoltagePins = channels[3:]

#set frequency of oscillation
freq = 3
halfT = 0.499/freq

#input desired opaque coordinates
x1,y1 = 1,1 #GPIO24 & GPIO21
x2,y2 = 2,2 #GPIO25 & GPIO22

try:
    while 1:
        
        if (x1 != x2) and (y1 != y2):
            GPIO.output([columnVoltagePins[x1],rowVoltagePins[y1],columnVoltagePins[x2],rowVoltagePins[y2]],[0,1,1,0])
            time.sleep(halfT)
        
            GPIO.output([columnVoltagePins[x1],rowVoltagePins[y1],columnVoltagePins[x2],rowVoltagePins[y2]],[1,0,0,1])
            time.sleep(halfT)
            
        elif x1 == x2:
            print("x1 = x2")
            
            GPIO.output([columnVoltagePins[x1],rowVoltagePins[y1],rowVoltagePins[y2]],[0,1,1])
            time.sleep(halfT)
        
            GPIO.output([columnVoltagePins[x1],rowVoltagePins[y1],rowVoltagePins[y2]],[1,0,0])
            time.sleep(halfT)
            
        elif y1 == y2:
            print("y1 = y2")
            
            GPIO.output([columnVoltagePins[x1],rowVoltagePins[y1],columnVoltagePins[x2]],[0,1,0])
            time.sleep(halfT)
        
            GPIO.output([columnVoltagePins[x1],rowVoltagePins[y1],columnVoltagePins[x2]],[1,0,1])
            time.sleep(halfT)
        
except KeyboardInterrupt:
    GPIO.cleanup()
