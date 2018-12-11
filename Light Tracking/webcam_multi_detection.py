#could there be something other than imutils that's just as flexible as picam? 

#all imports needed
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2


webcam = VideoStream(src=0).start()
time.sleep(2.0)

#initialize list of frames processed
frames = []
while True:
    frame = stream.read()
    

    #loop over frame sin webcam 

    #check to see if a key was pressed
    key = cv2.waitKey(1) & 0xFF
    #if 'q' was pressed, break
    if key == ord("q"):
        break

#cleanup
    cv2.destroyAllWindows()
    webcam.stop()

