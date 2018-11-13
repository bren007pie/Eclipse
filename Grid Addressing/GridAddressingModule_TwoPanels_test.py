import RPi.GPIO as GPIO
import time
import numpy as np

#grid function for display purposes
##def gridDisplay(row,column):
##    lengthRow = len(row)
##    lengthColumn = len(column)
##    grid = np.ones((lengthRow,lengthColumn))
##    
##    for i in range(lengthRow):
##        if row[i] == 1:
##            for j in range(lengthColumn):
##                if row[i] != column[j]:
##                    grid[i,j] = 0
##        else:
##            grid[i] = 0
##    print(grid)
##    print("\n")
##    return grid

#constants
numRows = 7
numColumns = 14
freq = 50
halfT = 0.5/freq

#input grid location
x1,y1 = 0,0
x2,y2 = 1,1

#"0" corresponds to voltage OFF and "1" corresponds to voltage ON
#initially the voltage is ON everywhere to signify complete transparency (PDLC)
rowVoltage = np.ones(numRows)
columnVoltage = np.ones(numColumns)
gridTotal = np.zeros((numRows,numColumns))

#coordinates are in separate columns and rows
if (x1 != x2) and (y1 != y2):
    i = 0
    while i < 100:
        rowVoltage[y1] = 0
        columnVoltage[x1] = 1
        columnVoltage[x2] = 0
        rowVoltage[y2] = 1
##        gridTotal += gridDisplay(rowVoltage,columnVoltage)
        time.sleep(halfT)
        
        columnVoltage[x1] = 0
        rowVoltage[y1] = 1
        rowVoltage[y2] = 0
        columnVoltage[x2] = 1
##        gridTotal += gridDisplay(rowVoltage,columnVoltage)
        time.sleep(halfT)
        
        i += 1

#x-coordinates are the same
elif x1 == x2:
    i = 0
    while i < 100:
        rowVoltage[y1] = 0
        rowVoltage[y2] = 0
        columnVoltage[x1] = 1
##        gridTotal += gridDisplay(rowVoltage,columnVoltage)
        time.sleep(halfT)
        
        columnVoltage[x1] = 0
        rowVoltage[y1] = 1
        rowVoltage[y2] = 1
##        gridTotal += gridDisplay(rowVoltage,columnVoltage)
        time.sleep(halfT)
        
        i += 1
        
#y-coordinates are the same
elif y1 == y2:
    i = 0
    while i < 100:
        rowVoltage[y1] = 0
        columnVoltage[x1] = 1
        columnVoltage[x2] = 1
##        gridTotal += gridDisplay(rowVoltage,columnVoltage)
        time.sleep(halfT)
        
        columnVoltage[x1] = 0
        columnVoltage[x2] = 0
        rowVoltage[y1] = 1
##        gridTotal += gridDisplay(rowVoltage,columnVoltage)
        time.sleep(halfT)
        
        i += 1
    
print("The desired opaque locations are", x1,y1, "and", x2,y2)
print("\nThe percentage(%) of time with a voltage across each coordinate:")
print(gridTotal*50/i)
