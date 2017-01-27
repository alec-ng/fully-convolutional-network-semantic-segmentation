import numpy as np
import sys, os
from PIL import Image

# SUMMARY 	: given the prediction, segments the human from the picture
# PARAM		: (origImg): path to original test image
# PARAM		: (maskImg): path to mask image
# RETURN 	: segmented image as an Image object
def segment(origImg, maskImg):
	origin = Image.open(origImg).convert('RGBA').getdata()
	mask = Image.open(maskImg).convert('RGBA').getdata()
	segImg = []
	total = origin.size[0] * origin.size[1]
	
	for count in range(0, total-1):
		if isBackground(mask[count]):
			segImg.append((255, 255, 255, 255))
		else:
			segImg.append(origin[count])

	origin = Image.open(origImg).convert('RGBA')
	origin.putdata(segImg)
	return origin


# PARAM: [pixel] RGBA format
def isBackground(pixel):
	return (pixel[0] == 255 and pixel[1] == 255 and pixel[2] == 255 and pixel[3] == 255)

