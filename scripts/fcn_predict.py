'''
Given an input image, runs it through the neural network to obtain
pixel by pixel classification

The output is saved in multiple formats
- labelled image, pixel classification
- black and white mask image
- segmented image

'''

import numpy as np
import sys, os, time, cv2
from PIL import Image
from segment import segment
from clahe import enhance
import caffe

# @PARAM	: (testImgPath) path to test image
# @PARAM	: (imgDir) path to images dir
#			: should have subdirs mask, segment, labels, npy
# @PARAM	: (clipSize) for CLAHE
# @PARAM	: (net) caffe net preloaded with trained model
# @RETURN 	: time taken for prediction
def predict(testImgPath, imgDir, clipSize, net):

	# get file name
	pathArr = testImgPath.split('/')
	tmpFileName = pathArr[len(pathArr) - 1]
	filename = os.path.splitext(tmpFileName)[0]

	# preprocess image
	processedImg = preprocessImg(testImgPath, clipSize)

	# reshape image to be put into data layer
	# shape for input (data blob is N x C x H x W)
	net.blobs['data'].reshape(1, *processedImg.shape)
	print('Predicting...')
	net.blobs['data'].data[...] = processedImg

	# run net and take argmax for prediction
	t = time.process_time()
	net.forward()
	elapsed_time = time.process_time() - t
	out = net.blobs['score'].data[0].argmax(axis=0)
	print("Prediction time: %.3f" % elapsed_time)

	print('Saving...')
	savePrediction(imgDir, out, filename, testImgPath)
	print('Done processing image ' + filename)

	return elapsed_time


# @SUMMARY 	: saves output of neural network into four different formats describe above
# @PARAM	: (imgDir) image target directory
# @PARAM	: (out) output of neural network
# @PARAM	: (filename) to save as
# @PARAM	: (testImgPath) for segmentation 
def savePrediction(imgDir, out, filename, testImgPath):
	# save as labelled image
	npyFile = filename + '.npy'		
	np.save(os.path.join(imgDir + 'npy/' + npyFile), out);

	Palette_base = [I for I in range(0, 256, int(255/3))];
	Palette = [(Palette_base [I], Palette_base [J], Palette_base [K]) for I in range (4) \
		for J in range ( 4) for K in range (4)];
	Colors = np.array(Palette, dtype = np.uint8)[out]
	labelledImage = Image.fromarray(Colors)
	labelledImage.save(os.path.join(imgDir + 'labelled/' + filename + '.png'), 'PNG')
	
	# save as mask
	out[out != 15] = 255 # mark anything not human as 0 (background)
	out[out == 15] = 0 # truth will be in black
	mask = out.astype('uint8')
	maskImPath = os.path.join(imgDir + 'mask/' + filename + '.png')
	Image.fromarray(mask).save(maskImPath, 'PNG')

	# save as segmented image
	segImg = segment(testImgPath, maskImPath)
	segImg.save(os.path.join(imgDir + 'seg/' + filename + '.png'), 'PNG')


# @SUMMARY	: loads image and preprocesses it to use for net
# @PARAM	: (imgPath) path to test image
# @RETURN	: returns a preprocessed Image
def preprocessImg(imgPath, clipSize):
	if clipSize != 0:
		im = enhance(imgPath, clipSize)
	else:
		im = cv2.imread(imgPath)
		im = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
		
	# switch to BGR, subtract mean
	in_ = np.array(im, dtype = np.float32)
	in_ = in_[:,:,::-1]
	in_ -= np.array((104.00698793,116.66876762,122.67891434))

	# make dims C x H x W for Caffe
	in_ = in_.transpose((2,0,1))

	return in_
