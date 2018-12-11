from imgproc import *
#create camera
camera = Camera(320,240)

#set viewer size (window size)
view = Viewer(cam.width, cam.height, "Blob finding")

while True:
    image = cam.grabImage()
    view.displayImage(image)
    

