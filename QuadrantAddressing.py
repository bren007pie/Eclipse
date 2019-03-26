import time
import adafruit_mcp230xx
import board
import busio
import digitalio
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(0)
channels = list(range(4,14)) + list(range(16,18))
GPIO.setup(channels, GPIO.OUT, initial = 0)

#set up "y-channels", i.e. the rows
ychannel1 = channels[0:3]
ychannel2 = channels[3:6]
ychannel3 = channels[6:9]
ychannel4 = channels[9:12]

#necessary initialization for the first column of each quadrant
GPIO.output(ychannel1[0],1)
GPIO.output(ychannel2[0],1)
GPIO.output(ychannel3[0],1)
GPIO.output(ychannel4[0],1)


i2c = busio.I2C(board.SCL, board.SDA)
mcp1 = adafruit_mcp230xx.MCP23017(i2c, address = 0x20)
mcp2 = adafruit_mcp230xx.MCP23017(i2c, address = 0x21)
mcp3 = adafruit_mcp230xx.MCP23017(i2c, address = 0x22)
mcp4 = adafruit_mcp230xx.MCP23017(i2c, address = 0x23)

quad1 = []
quad2 = []
quad3 = []
quad4 = []

for i in range(0,6):
    quad1.append(mcp1.get_pin(i))
    quad2.append(mcp2.get_pin(i))
    quad3.append(mcp3.get_pin(i))
    quad4.append(mcp4.get_pin(i))
    quad1[i].direction = digitalio.Direction.OUTPUT
    quad2[i].direction = digitalio.Direction.OUTPUT
    quad3[i].direction = digitalio.Direction.OUTPUT
    quad4[i].direction = digitalio.Direction.OUTPUT

#assuming parameters are being received from another part of the code
def blockTile(x,y,prev_x,prev_y):
    #check if new coords are the same as the last. If not then proceed.

    try:
        if x != prev_x or y !=prev_y:
            #assuming no error catching is necessary because x and y are
            #carefully prepared in another part of the code
            if x < 6:
                if y < 3:
                    for i in range(6): #range of 6 (0,1,2,3,4,5) should be enough?
                        quad1[i].value = 0
                    GPIO.output(ychannel1[y],1)
                    for i in range(3):
                        if i != y:
                            GPIO.output(ychannel1[i],0)
                    quad1[x].value = 1
                    print('done did it!!!')

                else:
                    for i in range(6): 
                        quad2[i].value = 0
                    GPIO.output(ychannel2[y-3],1)
                    for i in range(3):
                        if i+3 != y:
                            GPIO.output(ychannel2[i],0) #check with Stevie
                    quad2[x].value = 1
                    print('yah y33t')

            else:
                if y < 3:
                    for i in range(6): #range of 6 (0,1,2,3,4,5) should be enough?
                        quad3[i].value = 0 
                    GPIO.output(ychannel3[y],1)
                    for i in range(3):
                        if i != y:
                            GPIO.output(ychannel3[i],0)
                    quad3[x-6].value = 1
                    print('flicka da wrist boi')

                else:
                    for i in range(6): #range of 6 (0,1,2,3) should be enough?
                        quad4[i].value = 0
                    GPIO.output(ychannel4[y-3],1)
                    for i in range(3):
                        if i+3 != y:
                            GPIO.output(ychannel4[i],0)
                    quad4[x-6].value = 1
                    print('do whatcha daddy did')

        else:
            print('same tile as before!')
    except KeyboardInterrupt:
        GPIO.cleanup()
