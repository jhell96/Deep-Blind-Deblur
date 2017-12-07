import numpy as np
import cv2
import os

# path to video file
video = "video.mp4"

# make sure these directories exist
target_directory = 'targets'
blurred_directory = 'blurred'


def get_blurred_images(video_file_path, num_samples, blurring_param):
	'''
	Parameters:
		- video_file_path: path to .mp4 video file
		- num_samples: Number of samples to extract uniformly from the video
		- blurring param: Number of frames to average over

	Returns:
		- images
			. blurred images stored in a folder called "blurred"
			. target images stored in a folder called "targets"
	'''
	video_capture = cv2.VideoCapture(video_file_path)
	video_framecount = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
	frame_indicies = [i for i in range(video_framecount)]
	target_frame_indicies = [video_framecount // num_samples * i for i in range(num_samples + 1)]

	for i in range(len(target_frame_indicies)):
		video_capture.set(1, target_frame_indicies[i]);
		
		# GET THE TARGET FRAME
		# read the target frame in the video
		success, target_frame = video_capture.read()

		# store the target frame image as a jpg in target_directory
		cv2.imwrite(os.path.join(target_directory, "target_frame%d.jpg" % frame_indicies[i]), target_frame)

		if (target_frame_indicies[i] != target_frame_indicies[-1]):
			blurred_imgs = []
			for j in range(target_frame_indicies[i], target_frame_indicies[i+1], 1):
				# apply inverse gamma to each frame 
				# gamma function: g(x) = x^(1/2.2) 
				# inverse gamma function (according to wolfram alpha): g(x)^-1 = x^2.2
				video_capture.set(1, frame_indicies[j])
				success, layover_frame = video_capture.read()

				inv_gamma = lambda x: x**2.2
				inv_func = np.vectorize(inv_gamma)
				inv_frame = inv_func(layover_frame)
				print (j, inv_frame.shape)
				blurred_imgs.append(inv_frame)

			# iterate through all the inverse gamma'd frames, and take the summation and then the average
			print("*********")
			summation_of_frames= sum(blurred_imgs)/len(blurred_imgs)
			print(summation_of_frames.shape)

			# apply gamma function on this summation average
			gamma = lambda y: y**(1/2.2)
			func = np.vectorize(gamma)
			blurred_img = func(summation_of_frames)

			# store the blurred image as a jpg in blurred_directory
			cv2.imwrite(os.path.join(blurred_directory, "blurred_img%d.jpg" % frame_indicies[i]), blurred_img)

get_blurred_images(video, 10, 10)

















