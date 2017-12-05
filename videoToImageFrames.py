# reading in a video file and then spitting out images frame by frame
import cv2
vidcap = cv2.VideoCapture('video.mp4');
success, image = vidcap.read()
count = 0
success = True

while success:
	succes, image = vidcap.read()
	print('Read a new frame: ', success)
	cv2.imwrite("frame%d.jpg" % count, image) # save frame as a jpg file
	count += 1
