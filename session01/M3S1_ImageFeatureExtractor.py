#!/usr/bin/env python2
import cv2
import numpy as np
import os
import cPickle

# ImageFeatureExtractor class encapsulates the functionality of extracting
# features from images using several image descriptors.
# An instance of this class is expected to be configured first in order
# to impersonate a given image descriptor (such as SIFT).
# After configuration we can extractFeatures() from an image by passing its
# path by argument
class ImageFeatureExtractor:
	_slots__=['__configured', '__type', '__descriptor', '__hueHistogramBins','__recordData', '__useRecordedData', '__recordedDataName']

	#initialize vars
	def __init__(self, type, recordData = False, useRecordedData = False, recordedDataName = ''):
		print 'Hello'
		self.__configured = False
		self.__type = type
		self.__descriptor = None
		self.__recordData = recordData
		self.__useRecordedData = useRecordedData
		self.__recordedDataName = recordedDataName
		
		if self.__recordData:
			if not os.path.isdir('../../Databases/FeatureCache'):
				os.mkdir('../../Databases/FeatureCache')
			if not os.path.isdir('../../Databases/FeatureCache'+'/'+self.__recordedDataName):
				os.mkdir('../../Databases/FeatureCache'+'/'+self.__recordedDataName)
				
	def configureSIFT(self, numFeatures):
		#assert(not self.__configured)

		try:
			print 'cv3'
			self.__descriptor = cv2.xfeatures2d.SIFT_create(nfeatures=numFeatures)
		except:
			print 'cv2'
			self.__descriptor = cv2.SIFT(nfeatures=numFeatures)
			
		self.__configured = True
		self.__type = 'SIFT'

	def configureHOG(self):
		#assert(not self.__configured)

		winSize = (64,64)
		blockSize = (16,16)
		blockStride = (8,8)
		cellSize = (8,8)
		nbins = 9
		derivAperture = 1
		winSigma = 4.
		histogramNormType = 0
		L2HysThreshold = 2.0000000000000001e-01
		gammaCorrection = 0
		nlevels = 64
		self.__descriptor = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels)

		self.__configured = True
		self.__type = 'HOG'

	def configureHueHistogram(self, bins):
		#assert(not self.__configured)
		self.__hueHistogramBins = bins
		self.__configured = True
		self.__type = 'HUEHIST'

	def getType(self):
		return self.__type

	def isConfigured(self):
		return self.__configured

	# Extracts features of an image given a file path
	def extractFeatures(self, filename):
		#assert(self.__configured)
		kpt = None
		des = None
		path = "../../Databases/FeatureCache/" + self.__recordedDataName + "/" + os.path.basename(filename) + "_" + self.__type + ".p"
		
		if self.__useRecordedData:
			des = cPickle.load( open( path, "rb" ) )
			print str(len(des))+' extracted descriptors from '+path
		else:
			ima=cv2.imread(filename)
			gray=cv2.cvtColor(ima,cv2.COLOR_BGR2GRAY)
	
			if self.__type == 'SIFT':
				kpt,des = self.__descriptor.detectAndCompute(gray,None)
				print str(len(kpt))+' '+str(len(des))+' extracted keypoints and descriptors'
			elif self.__type == 'HOG':
				winStride = (8,8)
				padding = (8,8)
				locations = ((10,20),)
				des = self.__descriptor.compute(gray,winStride,padding,locations)     
			elif self.__type == 'HUEHIST':
				hsv = cv2.cvtColor(ima, cv2.COLOR_BGR2HSV)
				h,s,v = cv2.split(hsv)
				hist,bins = np.histogram(h.ravel(), self.__hueHistogramBins, density=True)
				des = np.array([hist])

		if self.__recordData:
			cPickle.dump( des, open(path, "w+b" ) )

		return kpt, des

