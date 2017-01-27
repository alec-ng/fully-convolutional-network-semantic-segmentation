from PIL import Image
from PIL import ExifTags
import os, sys

# @PARAM	: (sys.argv[1]) test img dir (non-scaled)
# @PARAM	: (sys.argv[2]) output img dir (scaled)
# @PARAM	: (sys.argv[3]) max dimension value, e.g. 800 (for 800px)
def resize():
	
	if len(sys.argv) != 4:
		raise Exception("Invalid number of input parameters.")

	inputDir = sys.argv[1]
	outputDir = sys.argv[2]
	maxDim = int(sys.argv[3])
	maxSize = (maxDim, maxDim)

	for imFileName in os.listdir(inputDir):

		# Portrait mode in camera is considered a rotation
		# Look at image's EXIF file to rotate if needed to preserve portrait mode
		image = Image.open(os.path.join(inputDir, imFileName))
		try:
		    for orientation in ExifTags.TAGS.keys(): 
		        if ExifTags.TAGS[orientation] == 'Orientation': 
		        	break 
		    exif = dict(image._getexif().items())

		    if exif[orientation] == 3 : 
		        image = image.rotate(180, expand=True)
		    elif exif[orientation] == 6 : 
		        image = image.rotate(270, expand=True)
		    elif exif[orientation] == 8 : 
		        image = image.rotate(90, expand=True)
		except:
		    print('Something went wrong with the Exif tags for ' + os.path.join(inputDir, imFileName))


		# only scale if we need to
		imDims = image.size
		if (imDims[0] > maxDim or imDims[1] > maxDim):
			# ANTIALIAS - high quality downsampling filter
			image.thumbnail(maxSize, Image.ANTIALIAS)

		image.save(os.path.join(outputDir, imFileName))

if __name__ == "__main__":
	resize();