import numpy as np
import cv2
from matplotlib import pyplot as plt

def func(path):    
    frame = cv2.imread(path)
    frame = cv2.resize(frame,(128,128))
    converted2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # Convert from RGB to HSV

    lowerBoundary = np.array([0,40,30],dtype="uint8")
    upperBoundary = np.array([43,255,254],dtype="uint8")
    skinMask = cv2.inRange(converted, lowerBoundary, upperBoundary)
    skinMask = cv2.addWeighted(skinMask,0.5,skinMask,0.5,0.0)
    
    skinMask = cv2.medianBlur(skinMask, 5)
    
    skin = cv2.bitwise_and(converted2, converted2, mask = skinMask)
 
    img2 = cv2.Canny(skin,60,60)

    
    #SURF
    surf = cv2.xfeatures2d.SURF_create()
    img2 = cv2.resize(img2,(256,256))
    kp, des = surf.detectAndCompute(img2,None)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(len(des))
    return des
