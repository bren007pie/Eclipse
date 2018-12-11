import time
import sys
import os

sys.path.append('./SunIOT/SDL_Pi_SI1145');

#import RPi.GPIO as GPIO

#set up GPIO using BCM numbering
#GPIO.setmode(GPIO.BCM)

#GPIO.setup(GPIO.OUT, initial=0)

import SDL_Pi_SI1145

sensor = SDL_Pi_SI1145.SDL_Pi_SI1145()

def readSunLight():        
	time.sleep(0.5)
	vis = sensor.readVisible()
	IR = sensor.readIR()
	UV = sensor.readUV()
	uvIndex = UV / 100.0
	print('Vis:' + str(vis))
	print('IR:' + str(IR))
	print('UV:' + str(UV))
        #print('UV:' + str(uvIndex))


while True:
    readSunLight();
