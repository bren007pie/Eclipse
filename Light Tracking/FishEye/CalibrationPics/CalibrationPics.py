import picamera
from time import sleep

camera = picamera.PiCamera()

camera.resolution = (640,480)

camera.start_preview()
sleep(5)

for i in range(25,51):
    sleep(5)
    camera.capture('/home/pi/Eclipse/FishEye/calibrate%s.jpg' % i)


camera.stop_preview()
