import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(0)

channels = list(range(2,20)) 
GPIO.setup(channels,GPIO.OUT,initial = 0)

T = 1.0

while True:
    time.sleep(T)        
    #this single output pin controls both the normally closed
    #and normally open connections on the SPDT.
    GPIO.output(channels[0],1)
    #setting to 0 means the normally closed (top connection) is on,
    #setting to 1 means the normally open (bottom connection) is on.

    time.sleep(T)
    GPIO.output(channels[0],0)

