import numpy as np
import cv2
import os 

#File Properties
filename = 'Tracking_Video.mp4'
frames_per_second = 24.0
my_res= '720p'

def change_res(cap, width, height):
	cap.set(3,width)
	cap.set(4,height)

STD_DIMENSIONS = {
	"480p": (640,480),
	"720p": (1280, 720),
	"1080p": (1920, 1080),
	"4k": (3840, 2160),
}

def get_dims(cap, res='1080p'):
	width, height = STD_DIMENSIONS['480p']
	if res in STD_DIMENSIONS:
		width, height = STD_DIMENSIONS[res]
	change_res(cap, width, height)
	return width, height


VIDEO_TYPE = {
	'avi': cv2.VideoWriter_fourcc(*'XVID'),
	'mp4': cv2.VideoWriter_fourcc(*'XVID'),
}

def get_video_type(filename):
	filename, ext = os.path.splitext(filename)
	if ext in VIDEO_TYPE:
		return VIDEO_TYPE[ext]
	return VIDEO_TYPE['mp4']


cap = cv2.VideoCapture(0)
dims = get_dims(cap, res=my_res)

video_type_cv2 = get_video_type(filename)

out = cv2.VideoWriter(filename, video_type_cv2, frames_per_second, dims)


while True:
	ret, frame = cap.read()

	#Lower and Upper bounds for color detection
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	lower_green = np.array([[[50,107,158]]])
	upper_green = np.array([[[93,213,255]]])

	lower_orange = np.array([[[0,142,140]]]) 
	upper_orange = np.array([[[38,255,255]]])

	lower_blue = np.array([[[95,95,119]]])
	upper_blue = np.array([[[110,178,255]]])

	#Masks specfically looking for green, orange, and blue colors
	mask1 = cv2.inRange(hsv, lower_green, upper_green)
	green_contours, thresh = cv2.findContours(mask1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	mask2 = cv2.inRange(hsv, lower_orange, upper_orange)
	orange_contours, thresh = cv2.findContours(mask2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	mask3 = cv2.inRange(hsv, lower_blue, upper_blue)
	blue_contours, thresh = cv2.findContours(mask3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


	#Green Ping Pong Ball Tracking
	for g_cnt in green_contours:
		g_area = cv2.contourArea(g_cnt)
		
		if g_area > 2000:
	
			x, y, w, h = cv2.boundingRect(g_cnt)
			cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255, 0), 4)
			cv2.putText(frame,'Green',(x,y+h), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0,0,0), 2, cv2.LINE_AA)

	#Orange Ping Pong Ball Tracking
	for o_cnt in orange_contours:
		o_area = cv2.contourArea(o_cnt)
		
		if o_area > 2000:
	
			x, y, w, h = cv2.boundingRect(o_cnt)
			cv2.rectangle(frame, (x,y), (x+w,y+h), (0,128, 255), 4)
			cv2.putText(frame,'Orange',(x,y+h), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0,0,0), 2, cv2.LINE_AA)


	#Blue Ping Pong Ball Tracking
	for b_cnt in blue_contours:
		b_area = cv2.contourArea(b_cnt)
		
		if b_area > 2000:
	
			x, y, w, h = cv2.boundingRect(b_cnt)
			cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0, 0), 4)
			cv2.putText(frame,'Blue',(x,y+h), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0,0,0), 2, cv2.LINE_AA)

	#Displays three indiviudal masks and resulting live video with object tracking 
	cv2.imshow('frame', frame)
	cv2.imshow('green mask', mask1)
	cv2.imshow('orange mask', mask2)
	cv2.imshow('blue mask', mask3)

	out.write(frame)

	if cv2.waitKey(1) == ord('q'):
		break

cap.release()
out.release()
cv2.destroyAllWindows()