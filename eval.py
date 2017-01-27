import sys, os, caffe
sys.path.insert(0, 'scripts/')
from fcn_predict import predict
from computeAccuracies import computeAccuracies
from accuracy import computeAccuracy

# PARAM 	: (testSetPath), path to dir containinig test/ and truth/
# PARAM 	: (clipSize), clipsize for CLAHE preprocessing step
def eval(testSetPath, clipSize):

	checkTestAndTruths(testSetPath)
	checkDirs(testSetPath)
	
	imgTestDir = os.path.join(testSetPath, 'test/')

	caffe.set_mode_cpu()

	printTitle("Loading caffe model")
	net = caffe.Net(
		  'src/deploy.prototxt'
		, 'src/fcn8s-heavy-pascal.caffemodel'
		, caffe.TEST)
	print("Model loaded.")

	testImgPaths = []
	for testImg in os.listdir(imgTestDir):
		testImgPaths.append(imgTestDir + testImg)
	testImgPaths.sort()

	printTitle("Generating predictions")
	totalPredictTime = 0.00
	for count in range(0, len(testImgPaths)):
		totalPredictTime += predict(testImgPaths[count], testSetPath, clipSize, net)

	printTitle("Average prediction time: %.3f" % (totalPredictTime / len(testImgPaths)))

	printTitle("Generating scores")
	segPath = os.path.join(testSetPath, 'seg/')
	truthPath = os.path.join(testSetPath, 'truth/')
	computeAccuracies(testSetPath, segPath, truthPath)

	print("eval.py has completed.")


# Make folders for seg, labelled and mask predictions
def checkDirs(testSetPath):
	segPath = os.path.join(testSetPath, 'seg/')
	labelledPath = os.path.join(testSetPath, 'labelled/')
	maskPath = os.path.join(testSetPath, 'mask/')
	npyPath = os.path.join(testSetPath, 'npy/')

	if (os.path.isdir(segPath) or os.path.isdir(labelledPath) 
		or os.path.isdir(maskPath) or os.path.isdir(npyPath)):

		raise Exception("Unclean test dir - delete all subdirs not test or truth")
	
	os.makedirs(segPath)
	os.makedirs(labelledPath)
	os.makedirs(maskPath)
	os.makedirs(npyPath)

def checkTestAndTruths(testSetPath):
	truths = []
	tests = []

	testPath = os.path.join(testSetPath, 'test/')
	truthPath = os.path.join(testSetPath, 'truth/')

	for filename in os.listdir(testPath):
		tests.append(os.path.splitext(filename)[0])
	for filename in os.listdir(truthPath):
		truths.append(os.path.splitext(filename)[0])

	if len(truths) != len(tests):
		raise Exception("You don't have the same number of test and truth images.")

	truths.sort()
	tests.sort()

	for count in range(0, len(truths)):
		if truths[count] != tests[count]:
			print(truths[count])
			print(tests[count])
			raise Exception("Test and truth images don't match, or are missing. \
				Please check your images.")

def printTitle(str):
	print("\n" + str)
	print("============================================================\n")

if __name__ == "__main__":
	if (len(sys.argv) != 3):
		raise Exception("Invalid number of params.")

	testSetPath = sys.argv[1]
	clipSize = float(sys.argv[2])
	eval(testSetPath, clipSize);