import numpy as np
from math import exp
import math
from skimage.measure import compare_ssim as SSIM
import cv2
import os
from PIL import Image

def PSNR(img1, img2):
	mse = np.mean( (img1 - img2) ** 2 )
	if mse == 0:
		return 100
	PIXEL_MAX = 255.0
	return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def scale_width(img, t):
	return cv2.resize(img,  dsize=(256,144), interpolation = Image.BICUBIC)


avgPSNR = 0.0
avgSSIM = 0.0
counter = 0

DIR = "results/experiment_name/test_latest/images"
REAL = "../../../data/images/gopro/all/testB"
files = os.listdir(REAL)

for i, f in enumerate(files):
	
	if (len(f.split('_')) > 2):	
	
		name = f.split('_target_')
		name = name[0] + '_blurred_' + name[1]
		ns = name.split('.')
		name = ns[0] + '_fake_B.' + 'png'
	else:
		name = f.split('.')[0] + '_fake_B.'+ 'png'

	im_target = cv2.imread(REAL + '/' + f)
	im_pred = cv2.imread(DIR + '/' + name)
	im_target = scale_width(im_target, 640)	

	if im_pred is not None and im_target is not None:
		counter = i
		im_pred = np.array(im_pred)
		im_target = np.array(im_target)

		avgPSNR += PSNR(im_pred, im_target)
		avgSSIM += SSIM(im_pred, im_target, multichannel=True)

avgPSNR /= counter
avgSSIM /= counter
print('PSNR = %f, SSIM = %f' % (avgPSNR, avgSSIM))


