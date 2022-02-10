import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while True:
	ret, frame = cap.read()
	width = int(cap.get(3))
	height = int(cap.get(4))

	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	lower_green = np.array([[[73,117,125]]])
	upper_green = np.array([[[97,255,250]]])

	lower_orange = np.array([[[0,102,190]]]) 
	upper_orange = np.array([[[22,255,255]]])

	lower_blue = np.array([[[95,94,158]]])
	upper_blue = np.array([[[171,255,255]]])


	mask1 = cv2.inRange(hsv, lower_green, upper_green)
	green_contours, thresh = cv2.findContours(mask1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	mask2 = cv2.inRange(hsv, lower_orange, upper_orange)
	orange_contours, thresh = cv2.findContours(mask2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	mask3 = cv2.inRange(hsv, lower_blue, upper_blue)
	blue_contours, thresh = cv2.findContours(mask3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


	#Green Ping Pong Ball Tracking
	for g_cnt in green_contours:
		g_area = cv2.contourArea(g_cnt)
		
		if g_area > 350:
	
			x, y, w, h = cv2.boundingRect(g_cnt)
			cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255, 0), 4)
			cv2.putText(frame,'Green',(x,y+h), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0,0,0), 2, cv2.LINE_AA)

	#Orange Ping Pong Ball Tracking
	for o_cnt in orange_contours:
		o_area = cv2.contourArea(o_cnt)
		
		if o_area > 350:
	
			x, y, w, h = cv2.boundingRect(o_cnt)
			cv2.rectangle(frame, (x,y), (x+w,y+h), (0,128, 255), 4)
			cv2.putText(frame,'Orange',(x,y+h), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0,0,0), 2, cv2.LINE_AA)


	#Blue Ping Pong Ball Tracking
	for b_cnt in blue_contours:
		b_area = cv2.contourArea(b_cnt)
		
		if b_area > 350:
	
			x, y, w, h = cv2.boundingRect(b_cnt)
			cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0, 0), 4)
			cv2.putText(frame,'Blue',(x,y+h), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0,0,0), 2, cv2.LINE_AA)


	cv2.imshow('frame', frame)
	cv2.imshow('green mask', mask1)
	cv2.imshow('orange mask', mask2)
	cv2.imshow('blue mask', mask3)
	

	if cv2.waitKey(1) == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()