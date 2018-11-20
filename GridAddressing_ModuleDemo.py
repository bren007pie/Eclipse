import RPi.GPIO as GPIO
import time
import numpy as np

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(0)

#list of pins to be used as outputs
channels = list(range(4,25))
GPIO.setup(channels, GPIO.OUT, initial = 1)

#list of pins corresponding to row and column voltage control
rowVoltagePins = channels[:7]
columnVoltagePins = channels[7:]

#set frequency of oscillation
freq = 1
halfT = 0.49/freq

#grid function for display purposes
def gridDisplay(row,column):
    lengthRow = len(row)
    lengthColumn = len(column)
    grid = np.ones((lengthRow,lengthColumn))
    
    for i in range(lengthRow):
        if row[i] == 1:
            for j in range(lengthColumn):
                if row[i] != column[j]:
                    grid[i,j] = 0
        else:
            grid[i] = 0
    print(grid)
    print("\n")
    return grid


def flicker2(X, Y, halfT):
    GPIO.output([X[0],Y[0]],[0,1])
    time.sleep(halfT)
    GPIO.output([X[0],Y[0]],[1,0])
    time.sleep(halfT)

def flicker3(X, Y, halfT):
    GPIO.output([X[0],Y[0],Y[1]],[0,1,1])
    time.sleep(halfT)
    GPIO.output([X[0],Y[0],Y[1]],[1,0,0])
    time.sleep(halfT)

def flicker4(X, Y, halfT, inPhase):
    if inPhase == True:
        GPIO.output([X[0],Y[0],X[1],Y[1]],[0,1,0,1])
        time.sleep(halfT)
        GPIO.output([X[0],Y[0],X[1],Y[1]],[1,0,1,0])
        time.sleep(halfT)
       
    elif inPhase == False:
        GPIO.output([X[0],Y[0],X[1],Y[1]],[0,1,1,0])
        time.sleep(halfT)
        GPIO.output([X[0],Y[0],X[1],Y[1]],[1,0,0,1])
        time.sleep(halfT)
        
def flicker6(X, Y, halfT):
    GPIO.output([X[0],Y[0],X[1],Y[1],X[2],Y[2]],[0,1,0,1,1,0])
    time.sleep(halfT)
    GPIO.output([X[0],Y[0],X[1],Y[1],X[2],Y[2]],[1,0,1,0,0,1])
    time.sleep(halfT)
    
def flicker8(X, Y, halfT, inPhase):
    if inPhase == True:
        GPIO.output([X[0],Y[0],X[1],Y[1],X[2],Y[2],X[3],Y[3]],[0,1,0,1,0,1,0,1])
        time.sleep(halfT)
        GPIO.output([X[0],Y[0],X[1],Y[1],X[2],Y[2],X[3],Y[3]],[1,0,1,0,1,0,1,0])
        time.sleep(halfT)
       
    elif inPhase == False:
        GPIO.output([X[0],Y[0],X[1],Y[1],X[2],Y[2],X[3],Y[3]],[0,1,0,1,1,0,1,0])
        time.sleep(halfT)
        GPIO.output([X[0],Y[0],X[1],Y[1],X[2],Y[2],X[3],Y[3]],[1,0,1,0,0,1,0,1])
        time.sleep(halfT)
    

#input desired opaque coordinates
x1,y1 = 0,0
x2,y2 = 1,1
x3,y3 = 2,1
x4,y4 = 3,2

try:
    while 1:
        
        if x2 == None:
            print("only one location")
            flicker2([columnVoltagePins[x1]],[rowVoltagePins[y1]], halfT)
        
        
        elif x3 == None:
            if (abs(x1-x2) == 1) and (abs(y1-y2) == 1):
                print("four-tile square")
                flicker4([columnVoltagePins[x1],columnVoltagePins[x2]], [rowVoltagePins[y1],rowVoltagePins[y2]], halfT, True)
            
            elif (x1 != x2) and (y1 != y2):
                print("two locations uncommon")
                flicker4([columnVoltagePins[x1],columnVoltagePins[x2]], [rowVoltagePins[y1],rowVoltagePins[y2]], halfT, False)
            
            elif x1 == x2:
                print("x1 = x2")
                flicker3([columnVoltagePins[x1]],[rowVoltagePins[y1],rowVoltagePins[y2]], halfT)
            
            elif y1 == y2:
                print("y1 = y2")
                flicker3([rowVoltagePins[y1]],[columnVoltagePins[x1],columnVoltagePins[x2]], halfT)
            
            
        #potentially incomplete: might require redundancy of which location makes up the four-tile square
        elif x4 == None:
            print("four-tile square + one extra location")
            if (abs(x1-x2) == 1) and (abs(y1-y2) == 1):
                flicker6([columnVoltagePins[x1],columnVoltagePins[x2],columnVoltagePins[x3]], [rowVoltagePins[y1],rowVoltagePins[y2],rowVoltagePins[y3]], halfT)
             
             
        #potentially incomplete: might require redundancy of which locations make up each four-tile square
        else:
            print("2 four-tile squares")
            if (abs(x1-x3) >= 2 and abs(x2-x4)) and (abs(y1-y3) >= 2 and abs(y2-y4)):
                flicker8([columnVoltagePins[x1],columnVoltagePins[x2],columnVoltagePins[x3],columnVoltagePins[x4]], [rowVoltagePins[y1],rowVoltagePins[y2],rowVoltagePins[y3],rowVoltagePins[y4]], halfT, False)
            
            elif ((x1 == x3) and (x2 == x4)) != ((y1 == y3) and (y2 == y4)):
                flicker8([columnVoltagePins[x1],columnVoltagePins[x2],columnVoltagePins[x3],columnVoltagePins[x4]], [rowVoltagePins[y1],rowVoltagePins[y2],rowVoltagePins[y3],rowVoltagePins[y4]], halfT, True)    
        
        
except KeyboardInterrupt:
    GPIO.cleanup()
