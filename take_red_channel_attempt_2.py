import cv2
 
cap = cv2.VideoCapture('dataset/s1/vid_s1_t2.mov')

success, img = cap.read()
fno = 0
while success:
    red_channel = img[:,:,2]
	# read next frame
	success, img = cap.read()