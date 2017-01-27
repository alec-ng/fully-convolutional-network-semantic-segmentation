'''
Implementation of recall and precision

Given a segmented test image and its ground truth image, compute
the algorithm's precision, recall, and f1 score

(Optional) returns back positive and negative instances in ground truth image

'''

from PIL import Image
import numpy, sys, os

# PARAM		: (truthPath): path to ground truth image
# PARAM		: (testPath): path to segmented test image
# RETURN	: tuple (recall, precision, f1 score, positive, negative)
def computeAccuracy(truthPath, testPath):
	truth = Image.open(truthPath).convert('RGBA').getdata()
	test = Image.open(testPath).convert('RGBA').getdata()

	if truth.size != test.size:
		raise TypeError('Inputted test and truth images do not match.')

	total = truth.size[0] * truth.size[1]
	tp = 0 # true positive
	tn = 0 # true negative
	fp = 0 # false positive
	fn = 0 # false negative
	pos = 0 # positive instances, ground truth pix = human
	neg = 0 # negative instances, ground truth pix = background

	for count in range(0, total - 1):
		truthPix = truth[count]
		testPix = test[count]

		if isHuman(truthPix) and isHuman(testPix):
			tp += 1
			pos += 1
		elif isHuman(truthPix) and isBackground(testPix):
			fn += 1
			pos += 1
		elif isBackground(truthPix) and isBackground(testPix):
			tn += 1
			neg += 1
		else: # isBackground(truthPix) and isHuman(testPix)
			fp += 1 
			neg += 1
			
	recall = tp / (tp + fn)
	precision = tp / (tp + fp)
	f1 = (2 * tp) / ( (2*tp) + fp + fn )

	return (recall, precision, f1, pos, neg)

# SUMMARY	: By our convention, a pixel is considered background if 
#				it's (255,255,255, 255) 
# PARAM		: [pixel] RGBA format
# RETURN 	: T/F
def isBackground(pixel):
	return (pixel[0] == 255 and pixel[1] == 255 and pixel[2] == 255 and pixel[3] == 255)

# SUMMARY	: By our convention, a pixel is considered background if 
#				it's not (255, 255, 255, 255)
# PARAM		: [pixel] RGBA format
# RETURN 	: T/F
def isHuman(pixel):
	return not(isBackground(pixel))