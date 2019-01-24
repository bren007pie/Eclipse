import serial 
import struct


#set up serial communication between pi's
ser = serial.Serial(
    port='/dev/serial0',
    baudrate = 9600, #default baudrate
    parity = serial.PARITY_NONE,#not ensuring accurate transmission
    stopbits = serial.STOPBITS_ONE,
    bytesize = serial.EIGHTBITS,
    timeout = 1
    )


#while True:
    #read ints of x,y, coordinates from other pi
#    x = ser.read()
#    print(x)
    #print(ord(x))
        
    

while True:
    
    #if ser.inWaiting():
    ser.flush()
#    boolean = ser.inWaiting()
#    print(boolean)
#    x = ser.read()
#    print(str(x))
    
    x_array = []
    for i in range(2):
        x_array.append(ser.read()) 
        print(x_array[i])
        
    x_packed = x_array[0] + x_array[1]
    x_unpacked = struct.unpack('h',x_packed)
    x = x_unpacked[0]
    print(x)
        
 
