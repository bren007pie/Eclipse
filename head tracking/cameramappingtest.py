import math as m

print(m.cos(m.pi/2)) #you can still put it in brackets in python 2!

#example eyecentres
eyecentres = [[100, 300], [200,300]] #left and right eye

dist = 20 #person is 20 cm away

#resolution = 1280 in x 720 in y
#FOV 60 degrees (assume both directions but test) #https://support.logitech.com/en_us/article/17556


#My way
eyedist = []
FOVx = 60*m.pi/180 #FOV in radians
thetax = FOVx/2
Resolutionx = 1280

objectplanex = dist*m.tan(thetax)
mapx = objectplanex/Resolutionx #gets cm/pixel at that distance


FOVy = 60*m.pi/180 #FOV in radians
thetay = FOVx/2
Resolutiony = 720

objectplaney = dist*m.tan(thetay)
mapy = objectplaney/Resolutiony #gets cm/pixel at that distance

for (x,y) in eyecentres:
    eyedist.append([mapx*x, mapy*y])

print(eyecentres)
print(mapx, mapy)
print(eyedist)


#Mario's way
    
    



