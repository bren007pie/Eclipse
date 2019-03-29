import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(0)

channels = list(range(4,14))+list(range(16,27))
GPIO.setup(channels, GPIO.OUT, initial = 0)

GPIO.output(19,1)

#Grid addressing matrix -> this is used to get the GPIO Pin coresponding to the matrix location of the real life grid

#0,0 is origin and is bottom left grid ellement

#               0,  1,  2,  3,  4,  5,  6 
GridMatrix0 = [ 4,  5,  6,  7,  8,  9, 10]
GridMatrix0 = [11, 12, 13, 16, 17, 18, 19] 

