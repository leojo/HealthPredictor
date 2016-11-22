import os
import inspect
import nibabel as nib
import numpy as np
import scipy as sp
import scipy.ndimage.interpolation as interpolation
import glob
import pickle
import math
from printProgress import printProgress

from sklearn.feature_extraction.image import grid_to_graph
from sklearn.cluster import AgglomerativeClustering

# The directory to store the precomputed features
featuresDir =  os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

# imgDir must be a string and maxValue must be an integer
# nBins is the number of "bins" for the histogram
def extractHistograms(imgDir, maxValue = 4000, nBins = -1, nPartitions = 1):
	if nBins == -1: nBins=maxValue

	# The number of different intensities per point of the histogram
	binSize = math.ceil((maxValue*1.)/nBins)
	imgPath = os.path.join(imgDir,"*")

	# This is the cache for the feature, used to make sure we do the heavy computations more often than neccesary
	outputFileName = os.path.join(featuresDir,"histograms_"+str(nBins)+"-"+str(maxValue)+"_"+imgDir.replace(os.sep,"-")+".feature")
	if os.path.isfile(outputFileName):
		save = open(outputFileName,'rb')
		histograms = pickle.load(save)
		save.close()
		return histograms

	# Fetch all directory listings of set_train and sort them on the image number
	allImageSrc = sorted(glob.glob(imgPath), key=extractImgNumber)
	histograms = []
	n_samples = len(allImageSrc);
	print "Found "+str(n_samples)+" images!"
	print "Preparing the data"
	printProgress(0, n_samples)
	for i in range(0,n_samples):
		img = nib.load(allImageSrc[i])
		imgData = img.get_data();

		# Count occurances of each intensity below the maxValue
		single_brain = np.array([[[[0]*nBins]*nPartitions]*nPartitions]*nPartitions)
		for x in range(imgData.shape[0]):
			for y in range(imgData.shape[1]):
				for z in range(imgData.shape[2]):
					val = imgData[x][y][z][0]
					partX = int((x*nPartitions)/imgData.shape[0])
					partY = int((y*nPartitions)/imgData.shape[1])
					partZ = int((z*nPartitions)/imgData.shape[2])
					if val < maxValue and val > 0:
						c = int(val/binSize)
						single_brain[partX][partY][partZ][c] += 1
		histograms.append(single_brain.flatten().tolist())
		printProgress(i+1, n_samples)

	print "\nStoring the features in "+outputFileName
	output = open(outputFileName,"wb")
	pickle.dump(histograms,output)
	output.close()
	print "Done"

	return histograms

def extractImgNumber(imgPath):
	imgName = imgPath.split(os.sep)[-1]
	imgNum = int(imgName.split('_')[-1][:-4])
	return imgNum
