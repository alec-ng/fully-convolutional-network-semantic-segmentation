'''
Given a set of test outputs and ground truths, generates a text file 
containing precision, recall and f1 scores for each test image.

Also computes the following statistical values for precision, recall and f1:
- high
- mean
- low
- standard deviation

'''

from PIL import Image
from accuracy import computeAccuracy
import numpy, sys, os, statistics

def computeAccuracies(testSetPath, testPath, truthPath):
	
	# Ensure that file names match up 
	checkTestsAndTruths(testPath, truthPath)

	print("Processing...")

	# compute scores
	filenames = []
	for filename in os.listdir(testPath):
		filenames.append(filename)

	# for each image, record precision, recall, f1
	txtFile = open(os.path.join(testSetPath + 'accuracy.txt'), 'w')
	txtFile.write(genHeader())
	recalls = []
	precisions = []
	f1s = []
	for count in range(0, len(filenames)):
		truthImg = os.path.join(truthPath  + filenames[count])
		testImg = os.path.join(testPath  + filenames[count])
		scoreTuple = computeAccuracy(truthImg, testImg)
		txtFile.write(genLine(filenames[count], scoreTuple))
		recalls.append(scoreTuple[0])
		precisions.append(scoreTuple[1])
		f1s.append(scoreTuple[2])

	# calculate high, low, mean, stddev metrics
	writeMetrics(txtFile, "RECALL", recalls)
	writeMetrics(txtFile, "PRECISION", precisions)
	writeMetrics(txtFile, "F1 SCORES", f1s)
	
	txtFile.close();
	print("Score.txt generated.")


# SUMMARY	: Write to a file high, low, mean, stddeev of a variable
# PARAM 	: (vals) list of values (numbers) of the variable
def writeMetrics(file, varName, vals):
	scoreTuple = (max(vals), min(vals), statistics.mean(vals), statistics.stdev(vals))
	file.write("\n" + varName + "\n")
	file.write("High: %.3f\nLow: %.3f\nMean: %.3f\nStdev: %.3f\n" % scoreTuple)

# SUMMARY	: Generates a header for the accuracy text file
# RETURN	: (str)
def genHeader():
	header = "FILENAME".ljust(60)
	header += "RECALL".ljust(12)
	header += "PRECISION".ljust(12)
	header += "F1".ljust(5) + "\n"
	header += "".rjust(len(header), '=')
	header += "\n"
	return header


# SUMMARY	: Generates a nicely formatted line of accuracies to insert in
#				the text file
# RETURN	: (str)
def genLine(filename, accuracyTuple):
	str = "\n"
	fileStr = filename.ljust(60)
	str += fileStr
	str += "%.3f".ljust(12) % accuracyTuple[0]
	str += "%.3f".ljust(12) % accuracyTuple[1]
	str += "%.3f\n" % accuracyTuple[2]
	return str


# SUMMARY	: Ensure test and truth dirs hold identical images.
#				Raises exception if this requirement fails.
# PARAM 	: (testPath)
# PARAM 	: (truthPath)
def checkTestsAndTruths(testPath, truthPath):
	tests = []
	truths = []
	
	for filename in os.listdir(testPath):
		tests.append(filename)
	for filename in os.listdir(truthPath):
		truths.append(filename)

	if len(truths) != len(tests):
		raise Exception("Mismatching number of test and truth images.")

	truths.sort()
	tests.sort()

	for count in range(0, len(truths)):
		if truths[count] != tests[count]:
			raise Exception("Test and truth images don't match - "
				+ truths[count] + ", " + tests[count])


if __name__ == "__main__":
	if len(sys.argv) != 3:
		raise Exception("Invalid number of params.")

	computeAccuracies(sys.argv[1], sys.argv[2])