import cv2
import numpy as np 

sampleNum = 0

uid = input('enter user id')

cam = cv2.VideoCapture(0)

while(True):
	ret,img = cam.read()	#ret ois used to find if the camera is providing the frames or not.....we can ignore this with "_"
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	x=20
	y=100
	w=300
	h=250
	sampleNum+=1
	#creates the gtreen rectangle
	cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
	img2=img[y:y+h,x:x+w]	


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

	cv2.imshow('closing_morphological transformation',closing)

	#saving address
	cv2.imwrite('data set1/'+str(uid)+'_'+str(sampleNum)+'.jpg',gray[y:y+h,x:x+w])
	cv2.imwrite('data set2/'+str(uid)+'_'+str(sampleNum)+'.jpg',closing[y:y+h,x:x+w])



	cv2.waitKey(100)	#there is a gap of 100 miliseconds between every frame caputured

	cv2.imshow('INPUT',img)

	cv2.waitKey(1)
	if(sampleNum>50):
		break
cam.release()
cam.destroyAllWindows()
