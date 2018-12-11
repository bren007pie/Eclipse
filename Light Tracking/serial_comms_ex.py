import time
import serial

#NOTES: GPIO pins available for UART on the Pi are: GPIO14&GPIO15

ser = serial.Serial(
    port='/dev/serial0',
    baudrate = 9600, #default baudrate
    parity = serial.PARITY_NONE,#not esnuring accurate transmission
    stopbits = serial.STOPBITS_ONE,
    bytesize = serial.EIGHTBITS,
    timeout = 1
    )

#write
while True:
    ser.write('Hello\n')
    time.sleep(1)

#read
while True:
    x = ser.readLine()
    
