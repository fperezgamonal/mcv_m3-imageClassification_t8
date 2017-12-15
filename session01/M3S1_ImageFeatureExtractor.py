#!/usr/bin/env python2
import cv2

# ImageFeatureExtractor class encapsulates the functionality of extracting
# features from images using several image descriptors.
# An instance of this class is expected to be configured first in order
# to impersonate a given image descriptor (such as SIFT).
# After configuration we can extractFeatures() from an image by passing its
# path by argument
class ImageFeatureExtractor:
	_slots__=['__configured','__type','__descriptor']

	#initialize vars
	def __init__(self):
		self.__configured = False
		self.__type = None
		self.__descriptor = None
			
	def configureSIFT(self, numFeatures):
		assert(not self.__configured)

		major = cv2.__version__.split(".")[0]

		#if major>=3:
		#    self.__descriptor = cv2.xfeatures2d.SIFT_create(=numFeatures)
		#else:
		self.__descriptor = cv2.SIFT(nfeatures=numFeatures)
			
		self.__configured = True
		self.__type = 'SIFT'
    
	def configureHOG(self):
		assert(not self.__configured)

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
		self.__descriptor = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
								histogramNormType,L2HysThreshold,gammaCorrection,nlevels)

		self.__configured = True
		self.__type = 'HOG'

	def getType(self):
		return self.__type

	def isConfigured(self):
		return self.__configured

	# Extracts features of an image given a file path
	def extractFeatures(self, filename):
		assert(self.__configured)

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

		return None, des