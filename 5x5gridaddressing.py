import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(0)

channels = list(range(2,20)) 
GPIO.setup(channels,GPIO.OUT,initial = 0)

T = 0.5
T2 = 0.05

#GPIO.output(channels[0],1)
#GPIO.output(channels[3],1)
#GPIO.output(channels[7],1)

while True:
   
    #GPIO.output(channels[:],0)
    #time.sleep(T)
    #GPIO.output(channels[:],0)
    #GPIO.output(channels[8],1)
    #time.sleep(T)
    #GPIO.output(channels[:],0)
    #GPIO.output(channels[9],1)
                
    time.sleep(T)
    GPIO.output(channels[:],0)
    time.sleep(T2)
    GPIO.output(channels[1],1)
    GPIO.output(channels[2],1)
    GPIO.output(channels[4],1)
    GPIO.output(channels[6],1)
    
    time.sleep(T)
    #GPIO.output(channels[:],0)
    time.sleep(T2)
    GPIO.output(channels[1],0)
    GPIO.output(channels[2],0)
    GPIO.output(channels[4],0)
    
    GPIO.output(channels[0],1)
    GPIO.output(channels[3],1)
    GPIO.output(channels[4],1)
    #GPIO.output(channels[6],1)

    time.sleep(T)
    #GPIO.output(channels[:],0)
    time.sleep(T2)
    GPIO.output(channels[3],0)
    GPIO.output(channels[4],0)
    
    #GPIO.output(channels[0],1)
    GPIO.output(channels[2],1)
    GPIO.output(channels[5],1)
    GPIO.output(channels[6],1)

    ######
    time.sleep(T)
    #GPIO.output(channels[:],0)
    time.sleep(T2)
    GPIO.output(channels[0],0)
    GPIO.output(channels[5],0)
    GPIO.output(channels[6],0)
    
    GPIO.output(channels[1],1)
    #GPIO.output(channels[2],1)
    GPIO.output(channels[4],1)
    GPIO.output(channels[7],1)

    time.sleep(T)
    #GPIO.output(channels[:],0)
    time.sleep(T2)
    GPIO.output(channels[1],0)
    GPIO.output(channels[2],0)
    
    
    GPIO.output(channels[0],1)
    GPIO.output(channels[3],1)
    #GPIO.output(channels[4],1)
    #GPIO.output(channels[7],1)

    time.sleep(T)
    #GPIO.output(channels[:],0)
    time.sleep(T2)
    GPIO.output(channels[3],0)
    GPIO.output(channels[4],0)
    
    #GPIO.output(channels[0],1)
    GPIO.output(channels[2],1)
    GPIO.output(channels[5],1)
    #GPIO.output(channels[7],1)

    ######
    time.sleep(T)
    #GPIO.output(channels[:],0)
    time.sleep(T2)
    GPIO.output(channels[0],0)
    GPIO.output(channels[5],0)
    GPIO.output(channels[7],0)
    
    GPIO.output(channels[1],1)
    #GPIO.output(channels[2],1)
    GPIO.output(channels[4],1)
    GPIO.output(channels[8],1)

    time.sleep(T)
    #GPIO.output(channels[:],0)
    time.sleep(T2)
    GPIO.output(channels[1],0)
    GPIO.output(channels[2],0)
    
    
    GPIO.output(channels[0],1)
    GPIO.output(channels[3],1)
    #GPIO.output(channels[4],1)
    #GPIO.output(channels[8],1)

    time.sleep(T)
    #GPIO.output(channels[:],0)
    time.sleep(T2)
    GPIO.output(channels[3],0)
    GPIO.output(channels[4],0)
    
    #GPIO.output(channels[0],1)
    GPIO.output(channels[2],1)
    GPIO.output(channels[5],1)
    #GPIO.output(channels[8],1)

    ######
    time.sleep(T)
    #GPIO.output(channels[:],0)
    time.sleep(T2)
    GPIO.output(channels[0],0)
    GPIO.output(channels[5],0)
    GPIO.output(channels[8],0)
    
    GPIO.output(channels[1],1)
    #GPIO.output(channels[2],1)
    GPIO.output(channels[4],1)
    GPIO.output(channels[9],1)

    time.sleep(T)
    #GPIO.output(channels[:],0)
    time.sleep(T2)
    GPIO.output(channels[1],0)
    GPIO.output(channels[2],0)
    
    GPIO.output(channels[0],1)
    GPIO.output(channels[3],1)
    #GPIO.output(channels[4],1)
    #GPIO.output(channels[9],1)

    time.sleep(T)
    #GPIO.output(channels[:],0)
    time.sleep(T2)
    
    GPIO.output(channels[3],0)
    GPIO.output(channels[4],0)
    
    #GPIO.output(channels[0],1)
    GPIO.output(channels[2],1)
    GPIO.output(channels[5],1)
    #GPIO.output(channels[9],1)


