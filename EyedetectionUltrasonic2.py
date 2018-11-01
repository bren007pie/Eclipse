#sourced from: https://www.instructables.com/id/Face-and-Eye-Detection-With-Raspberry-Pi-Zero-and-/

#Imports and libraries
import numpy as np
import cv2
import os
from imutils.video import FPS
#these files
import RPi.GPIO as GPIO
import time

#Sets up Camera and OpenCV and FPS
os.chdir("/usr/local/share/OpenCV/haarcascades/")
face_cascade = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_eye.xml')
cap = cv2.VideoCapture(0)
fps = FPS().start() #defines the FPS object


#Ultrasonic Setup

#This disables the warning about channel being in use
GPIO.setwarnings(False)

#GPIO Mode (BOARD / BCM)
GPIO.setmode(GPIO.BCM)
 
#set GPIO Pins
GPIO_TRIGGER = 18
GPIO_ECHO = 24
 
#set GPIO direction (IN / OUT)
GPIO.setup(GPIO_TRIGGER, GPIO.OUT)
GPIO.setup(GPIO_ECHO, GPIO.IN)

def distance():
    # set Trigger to HIGH
    GPIO.output(GPIO_TRIGGER, True)
 
    # set Trigger after 0.01ms to LOW
    time.sleep(0.00001)
    GPIO.output(GPIO_TRIGGER, False)
 
    StartTime = time.time()
    StopTime = time.time()
 
    # save StartTime
    while GPIO.input(GPIO_ECHO) == 0:
        StartTime = time.time()
 
    # save time of arrival
    while GPIO.input(GPIO_ECHO) == 1:
        StopTime = time.time()
 
    # time difference between start and arrival
    TimeElapsed = StopTime - StartTime
    # multiply with the sonic speed (34300 cm/s)
    # and divide by 2, because there and back
    distance = (TimeElapsed * 34300) / 2
 
    return distance

#main loop

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    dist = distance()


    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),1) #draws the rectangle around the corners in blue to the image object
        cv2.circle(img, (int(x+w/2),int(y+h/2)), 3, (255,0,0),2) #draws a blue circle of radius 3 in the centre of the face to the image object. Pixel reference need to be an integer
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        #print("Head Centre:", (x+w/2, y+h/2)) #prints the middle of the head location, will want to smooth values out later
       
        
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),1)
            cv2.circle(img, (int(x+ex+ew/2),int(y+ey+eh/2)), 2, (0,255,0),1) #draws a green circle of radius 3 in the centre of the eye to the image object. Pixel reference need to be an integer.
            #all eye dimensions are with reference to the face so need to add the face coord to each one 
            print("Eye Position:",(x+ex+ew/2, y+ey+eh/2,"%.1f cm" % dist)) #prints the middle of the eye location
          
            #will need to smooth or average out these values after
    cv2.imshow('img',img) #img reference is 0,0 in top left
    k = cv2.waitKey(30) & 0xff
    if k == 27: #press escape key to escape
        break
    fps.update() #updates the fps object each time the detection runs

fps.stop() #stops the fps counting
print("[INFO] approx FPS: {:.2f}".format(fps.fps()))
#cleans up
cap.release()
cv2.destroyAllWindows()

#if getting an error run:
#sudo modprobe bcm2835-v4l2
#before running the script
