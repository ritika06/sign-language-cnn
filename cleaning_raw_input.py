import numpy as np
import cv2

cam = cv2.VideoCapture(0)

mog_var = cv2.createBackgroundSubtractorMOG2() 	#used of MOG background reduction technique otherwise comment this 

while(True):
	ret,img= cam.read()
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

	x=20
	y=100
	w=300
	h=250

	img2=img[y:y+h,x:x+w]
	
	
	cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

	cv2.imshow("original",img)



	#thresholding
	ret, threshold = cv2.threshold(img2,12,255,cv2.THRESH_BINARY)
	grayscaled = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
	ret, threshold2 = cv2.threshold(grayscaled, 100,255,cv2.THRESH_BINARY)
	gaus = cv2.adaptiveThreshold(grayscaled, 255 ,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,115,1)

	cv2.imshow("adaptive threshold", gaus)



	#edge detection(canny edge detection)
	edges = cv2.Canny(img2 , 100, 200)

	cv2.imshow("Edges", edges)



	#MOG background reduction
	fgmask = mog_var.apply(img2)

	cv2.imshow('MOG background reduction', fgmask)


	#morphological transformation 
	hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

	lower_skin = np.array([0,40,30])
	upper_skin = np.array([43,255,254])

	mask = cv2.inRange(hsv, lower_skin, upper_skin)
	res = cv2.bitwise_and(img2, img2, mask = mask)

	kernal = np.ones((5,5), np.uint8)
	erosion = cv2.erode(mask, kernal, iterations =1)
	dilation = cv2.dilate(mask, kernal, iterations =1)

	opening = cv2.morphologyEx(mask , cv2.MORPH_OPEN, kernal)
	closing = cv2.morphologyEx(mask , cv2.MORPH_CLOSE, kernal)

	##cv2.imshow('erosion_morphological transformation',erosion)
	##cv2.imshow('dilation_morphological transformation',dilation)
	##cv2.imshow('opening_morphological transformation',opening)
	cv2.imshow('closing_morphological transformation',closing)
  

	if(cv2.waitKey(1)==ord('q')):
		break

cam.release()
cv2.destroyAllWindows()