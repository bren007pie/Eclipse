import math

#constants and variables
xCam1 = 35
yCam2 = 35
dSensorToWindshield = 30
acceptableError = 3

#outputs from Light Tracking (in deg)
alpha1 = 5
phi1 = 3
alpha2 = -4
phi2 = -34

#outputs from Driver Tracking (in cm)
horizontalDisplacement = 0
verticalDisplacement = 0
depthSensorToDriver = 20

#loop
while 1:
    x1 = y1 = x2 = y2 = x3 = y3 = x4 = y4 = None
    
    dDriver = dSensorToWindshield + depthSensorToDriver
    xDriver = xCam1 + horizontalDisplacement
    yDriver = yCam2 + verticalDisplacement
    
    if type(alpha1) != type(None):
        xProjection1 = dDriver*math.tan(math.radians(alpha1)) + xDriver
        yProjection1 = dDriver*math.tan(math.radians(phi1)) + yDriver
        
        column1 = int(xProjection1/10)
        row1 = int(yProjection1/10)
        
        xError1 = xProjection1 - (10*column1 + 5)
        yError1 = yProjection1 - (10*row1 + 5)
        
        x1 = column1
        y1 = row1
        
        if type(alpha2) != type(None):
            xProjection2 = dDriver*math.tan(math.radians(alpha2)) + xDriver
            yProjection2 = dDriver*math.tan(math.radians(phi2)) + yDriver
            
            column2 = int(xProjection2/10)
            row2 = int(yProjection2/10)
            
            xError2 = xProjection2 - (10*column2 + 5)
            yError2 = yProjection2 - (10*row2 + 5)
            
            x2 = column2
            y2 = row2
    
        #do stuff depending on the size of the error
        if (abs(xError1) > acceptableError) or (abs(yError1) > acceptableError):
            if xError1 < -acceptableError:
                x2 = column1 - 1
            elif xError1 > acceptableError:
                x2 = column1 + 1
            else:
                x2 = column1
            
            if yError1 < -acceptableError:
                y2 = row1 - 1
            elif yError1 > acceptableError:
                y2 = row1 + 1
            else:
                y2 = row1
            
            if type(alpha2) != type(None):
                x3 = column2
                y3 = row2
                
                if (abs(xError2) > acceptableError) or (abs(yError2) > acceptableError):
                    if xError2 < -acceptableError:
                        x4 = column2 - 1
                    elif xError2 > acceptableError:
                        x4 = column2 + 1
                    else:
                        x4 = column2
            
                    if yError2 < -acceptableError:
                        y4 = row2 - 1
                    elif yError2 > acceptableError:
                        y4 = row2 + 1
                    else:
                        y4 = row2
    
    
    #break is for print statement
    break

print(column1, xProjection1, xError1, " ",row1, yProjection1, yError1)
print(column2, xProjection2, xError2, " ",row2, yProjection2, yError2)