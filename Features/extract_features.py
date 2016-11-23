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

def extractCompleteBrain(imgDir):
	imgPath = os.path.join(imgDir,"*")
	# This is the cache for the feature, used to make sure we do the heavy computations more often than neccesary
	outputFileName = os.path.join(featuresDir,"complete_"+imgDir.replace(os.sep,"-")+".feature")
	if os.path.isfile(outputFileName):
		save = open(outputFileName,'rb')
		data = pickle.load(save)
		save.close()
		return data
	allImageSrc = sorted(glob.glob(imgPath), key=extractImgNumber)
	data = []
	n_samples = len(allImageSrc);
	print "Found "+str(n_samples)+" images!"
	print "Preparing the data"
	for i in range(0,n_samples):
		img = nib.load(allImageSrc[i])
		imgData = img.get_data();
		single_brain = imgData.flatten().tolist()
		data.append(single_brain)
		printProgress(i+1, n_samples)
	print "\n!!!!!NOT Storing the features !!!!"
	#output = open(outputFileName,"wb")
	#pickle.dump(data,output)
	#output.close()
	#print "Done"
	return data

def extractBrainPart(imgDir,n_divisions=3,x_part=0,y_part=0,z_part=0):
	imgPath = os.path.join(imgDir,"*")
	# This is the cache for the feature, used to make sure we do the heavy computations more often than neccesary
	outputFileName = os.path.join(featuresDir,"brainpart_"+str(n_divisions)+"_"+str(x_part)+"_"+str(y_part)+"_"+str(z_part)+"_"+imgDir.replace(os.sep,"-")+".feature")
	if os.path.isfile(outputFileName):
		save = open(outputFileName,'rb')
		data = pickle.load(save)
		save.close()
		return data
	allImageSrc = sorted(glob.glob(imgPath), key=extractImgNumber)
	data = []
	n_samples = len(allImageSrc);
	print "Found "+str(n_samples)+" images!"
	print "Preparing the data"
	for i in range(0,n_samples):
		img = nib.load(allImageSrc[i])
		imgData = img.get_data();
		single_brain = []
		for x in range((x_part)*imgData.shape[0]/n_divisions,(x_part+1)*imgData.shape[0]/n_divisions):
			for y in range((y_part)*imgData.shape[1]/n_divisions,(y_part+1)*imgData.shape[1]/n_divisions):
				for z in range((z_part)*imgData.shape[2]/n_divisions,(z_part+1)*imgData.shape[2]/n_divisions):
					single_brain.append(imgData[x][y][z][0])
		data.append(single_brain)
		printProgress(i+1, n_samples)
	print "\n!!!!!NOT Storing the features !!!!"
	#output = open(outputFileName,"wb")
	#pickle.dump(data,output)
	#output.close()
	#print "Done"
	return data

# This was an attempt at a more sophisticated feature using agglomerative clustering to define "colors"
# and then taking a histogram of those color. This did not prove to give better results.
def extractHierarchicalClusters(imgDirFullPath, n_clusters=10, ignoreCache=False, scale=0.10):
	imgPath = os.path.join(imgDirFullPath,"*")

	outputFileName = os.path.join(featuresDir,"hierarchicalclusters_"+str(n_clusters)+"_"+str(scale)+"_"+imgDirFullPath.replace(os.sep,"-")+".feature")
	
	if os.path.isfile(outputFileName) and not ignoreCache:
		print("Found!")
		save = open(outputFileName,'rb')
		clusters = pickle.load(save)
		save.close()
		return clusters
	
	# Fetch all directory listings of set_train
	allImageSrc = sorted(glob.glob(imgPath), key=extractImgNumber)
	clusters = []
	n_samples = len(allImageSrc);
	print "Found "+str(n_samples)+" images!"
	print "Preparing the data"
	printProgress(0, n_samples)
	for i in range(0,n_samples):
		hist = [0]*n_clusters
		total_intensities = [0]*n_clusters
		img = nib.load(allImageSrc[i])
		imgData_original = np.asarray(img.get_data()[:,:,:,0])
		# Resize to 10% of original size for faster processing
		imgData_resized = sp.ndimage.interpolation.zoom(imgData_original,scale)
		imgData = np.reshape(imgData_resized,(-1,1))

		connectivity = grid_to_graph(*imgData_resized.shape)
		ward = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward',connectivity=connectivity)
		labels = ward.fit_predict(imgData).flatten().tolist()

		for j, lab in enumerate(labels):
			intensity = imgData[j][0]
			total_intensities[lab] += intensity
			hist[lab] += 1

		avg_intensity = np.asarray(total_intensities)*1./np.asarray(hist)
		avg_intensity = avg_intensity.flatten().tolist()
		avg_intensity, hist = zip(*sorted(zip(avg_intensity,hist)))
		
		clusters.append(hist)
		printProgress(i+1, n_samples)
	print "Done"
	print "\nStoring the features in "+outputFileName

	output = open(outputFileName,"wb")
	pickle.dump(clusters,output)
	output.close()
	print "Done"
	return clusters

# The AgglomerativeClusters approach was suffering from the severely reduced "resolution" of the images
# and this was an attempt to improve on that by only looking at one slice of the image instead of reducing the
# "resolution". This too was unsuccessful.
def extractHierarchicalClustersSingleSlice(imgDirFullPath, n_clusters=10, ignoreCache=False):
	imgPath = os.path.join(imgDirFullPath,"*")

	outputFileName = os.path.join(featuresDir,"hierarchicalclusterssingleslice_"+str(n_clusters)+"_"+imgDirFullPath.replace(os.sep,"-")+".feature")
	if os.path.isfile(outputFileName) and not ignoreCache:
		save = open(outputFileName,'rb')
		clusters = pickle.load(save)
		save.close()
		return clusters

	# Fetch all directory listings of set_train
	allImageSrc = sorted(glob.glob(imgPath), key=extractImgNumber)
	clusters = []
	n_samples = len(allImageSrc);
	print "Found "+str(n_samples)+" images!"
	print "Preparing the data"
	printProgress(0, n_samples)
	for i in range(0,n_samples):
		hist = [0]*n_clusters
		total_intensities = [0]*n_clusters
		img = nib.load(allImageSrc[i])
		imgData_original = np.asarray(img.get_data()[:,:,:,0])
		brainSlice = imgData_original[:,:,imgData_original.shape[2]/2]
		imgData = np.reshape(brainSlice,(-1,1))

		connectivity = grid_to_graph(*brainSlice.shape)
		ward = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward',connectivity=connectivity)
		labels = ward.fit_predict(imgData).flatten().tolist()

		for j, lab in enumerate(labels):
			intensity = imgData[j][0]
			total_intensities[lab] += intensity
			hist[lab] += 1
		avg_intensity = np.asarray(total_intensities)*1./np.asarray(hist)
		avg_intensity = avg_intensity.flatten().tolist()
		avg_intensity, hist = zip(*sorted(zip(avg_intensity,hist)))

		clusters.append(hist)
		printProgress(i+1, n_samples)
	print "Done"
	print "\nStoring the features in "+outputFileName

	output = open(outputFileName,"wb")
	pickle.dump(clusters,output)
	output.close()
	print "Done"
	return clusters

def extractImgNumber(imgPath):
	imgName = imgPath.split(os.sep)[-1]
	imgNum = int(imgName.split('_')[-1][:-4])
	return imgNum
