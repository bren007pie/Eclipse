import serial 
import struct


#set up serial communication between pi's
ser = serial.Serial(
    port='/dev/serial0',
    baudrate = 9600, #default baudrate
    parity = serial.PARITY_NONE,#not esnuring accurate transmission
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
    x_array = []
    for i in range(2):
        x_array.append(ser.read()) 
        print(x_array[i])
        
    x_packed = x_array[0] + x_array[1]
    x_unpacked = struct.unpack('h',x_packed)
    x = x_unpacked[0]
    print(x)
    #x_packed_slice = x_packed[:2]
    #print(x_packed_slice)
    #x_unpacked = struct.unpack('h',x_packed_slice)
    #x = x_unpacked[0]        
    #print(x)
    #print(float(x_unpacked[0]))
  
    #str_total= ""
    #str_list = []
    #a = bytearray
    #for i in range(2):
        #str_list.append(ser.read())
        #print(str_list)
        #byte_read = ser.read()
        #a.append(byte_read)
        #print(byte_read)
        #sliced_byte = byte_read[2:]
        #print(sliced_byte)
        
        #str_i = str(byte_read)
        #str_total += str_i#[2:]
        #str_total += str_list[i](2) #add only the part of the string after the first two characters
    #now have concatenated string of appropriate char's
    #need to add the " b' " in front to unpack
    #str_total = "b'" + str_total
    
    #join the bytes
    #conjoined_byte = b"".join([a[0],a[1]])
    #print(conjoined_byte)

    #convert back to byte
    #str_total_encoded = str_total.encode()#bytes(str_total)
    #now unpack
    #print(str_total)
    #print(str_total_encoded)

    #x = struct.unpack('h',str_total_encoded)
    #print(x)
        
 
