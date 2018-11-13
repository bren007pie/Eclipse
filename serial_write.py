import serial
import time
import struct


ser = serial.Serial(
    port = '/dev/serial0',
    baudrate = 9600,
    parity = serial.PARITY_NONE,
    stopbits = serial.STOPBITS_ONE,
    bytesize = serial.EIGHTBITS,
    timeout = 1
)

while True:
    #ser.write(str.encode('hello\n'))
    #x = str.encode(encoding='utf-8',chr(400))
    #ser.write(x)
    #time.sleep(0.1)
    #x = struct.pack('f',400.0)
    x = struct.pack('h',420)
    ser.write(x)
    #y = struct.unpack('f',x)
    y = struct.unpack('h',x)
    print(x)
    print(y)
    print(y[0])
    #print(ord('\x90'))
    #print(ord('\x01'))
    time.sleep(0.1)



    
