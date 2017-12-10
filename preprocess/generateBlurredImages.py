import numpy as np
import cv2
import os

def get_blurred_images(video_file_path, average_frames, num_samples):
	'''
	Parameters:
		- video_file_path: path to .mp4 video file
		- num_samples: Number of samples to extract uniformly from the video
		- average_frames: Number of frames to average over

	Returns:
		- images
			. blurred images stored in a folder called "blurred"
			. target images stored in a folder called "targets"
	'''
	video_capture = cv2.VideoCapture(video_file_path)
	video_framecount = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
	target_frame_indicies = np.random.choice(video_framecount - average_frames - 1, num_samples)


	for sample_number, target_index in enumerate(target_frame_indicies):

		blurred_imgs = []
		for frame_index in range(target_index, target_index + average_frames):
			# apply inverse gamma to each frame 
			# gamma function: g(x) = x^(1/2.2) 
			# inverse gamma function (according to wolfram alpha): g(x)^-1 = x^2.2
			video_capture.set(1, frame_index)
			success, layover_frame = video_capture.read()

			inv_gamma = lambda x: x**2.2
			inv_func = np.vectorize(inv_gamma)
			inv_frame = inv_func(layover_frame)
			blurred_imgs.append(inv_frame)	

		# iterate through all the inverse gamma'd frames, and take the summation and then the average
		summation_of_frames= sum(blurred_imgs)/len(blurred_imgs)

		# apply gamma function on this summation average
		gamma = lambda y: y**(1/2.2)
		func = np.vectorize(gamma)
		blurred_img = func(summation_of_frames)


		video_capture.set(1, target_index);
		
		# reading in the target frame
		# read the target frame in the video
		success, target_frame = video_capture.read()

		video_name = os.path.basename(video_file_path).split('.')[0]

		# store the target frame image as a jpg in target_directory
		cv2.imwrite(os.path.join(target_directory, video_name + "_target_%d.jpg" % sample_number), target_frame)

		# store the blurred image as a jpg in blurred_directory
		cv2.imwrite(os.path.join(blurred_directory, video_name + "_blurred_%d.jpg" % sample_number), blurred_img)

if __name__ == '__main__':

	# path containing vids
	videos_path = "/home/jhellerstein96/disk/videos-flat"

	# make sure these directories exist
	target_directory = '/home/jhellerstein96/disk/images/ytb/targets'
	blurred_directory = '/home/jhellerstein96/disk/images/ytb/blurred'

	num_samples_per_vid = 100
	blur_amount = 8

	total = len(os.listdir(videos_path))
	for i, vid_name in enumerate(os.listdir(videos_path)):
		path = os.path.join(videos_path, vid_name)
		print(path, i, "of", total)
		get_blurred_images(path, blur_amount, num_samples_per_vid)
















