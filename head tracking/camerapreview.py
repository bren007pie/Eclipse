from picamera import PiCamera
from time import sleep

#PicName = input('Enter the name of the .jpg to be taken:')

camera = PiCamera()
camera.start_preview()
sleep(180)

#camera.capture('/home/pi/eclipse/head tracking/' + PicName + '.jpg')
camera.stop_preview()
