#!/usr/bin/env python2
import cv2
import numpy as np
#from sklearn.lda import LDA
from sklearn.decomposition import PCA

# ImageFeatureExtractor class encapsulates the functionality of extracting
# features from images using several image descriptors.
# An instance of this class is expected to be configured first in order
# to impersonate a given image descriptor (such as SIFT).
# After configuration we can extractFeatures() from an image by passing its
# path by argument
class ImageFeatureExtractor:
	_slots__=['__configured', '__type', '__descriptor', '__hueHistogramBins']

	#initialize vars
	def __init__(self, type):
		print 'Initializing environment...'
		self.__configured = False
		self.__type = type
		self.__descriptor = None
		
	def configureSIFT(self, numFeatures):
		#assert(not self.__configured)
		
		try:
			print 'Using opencv: cv3'
			self.__descriptor = cv2.xfeatures2d.SIFT_create(nfeatures=numFeatures)
		except:
			print 'Using opencv: cv2'
			self.__descriptor = cv2.SIFT(nfeatures=numFeatures)	
		
		self.__configured = True
		self.__type = 'SIFT'

	def configureDenseSIFT(self, step, scales):
		self.__DSstep = step
		self.__DSscales = scales
		
		try:
			print 'Using opencv: cv3'
			self.__descriptor = cv2.xfeatures2d.SIFT_create()
		except:
			print 'Using opencv: cv2'
			self.__descriptor = cv2.SIFT()	
			
		self.__configured = True
		self.__type = 'DenseSIFT'

	def configureHOG(self, scale):#numBins=9):
		#assert(not self.__configured)
		
		windowSize = 256
		winSize = (windowSize,windowSize)
		blockSize = (windowSize/(2*scale),windowSize/(2*scale))
		blockStride = (windowSize/(4*scale),windowSize/(4*scale))
		cellSize = (windowSize/(4*scale),windowSize/(4*scale))
		nbins = 9
		derivAperture = 1
		winSigma = -1.
		histogramNormType = 0
		L2HysThreshold = 0.2
		gammaCorrection = 1
		nlevels = windowSize
		signedGradients = True

		self.__descriptor = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels,signedGradients)

		self.__configured = True
		self.__type = 'HOG'
		self.__paramStr = "???"

	def configureHueHistogram(self, bins):
		#assert(not self.__configured)
		self.__hueHistogramBins = bins
		self.__configured = True
		self.__type = 'HUEHIST'

	def getType(self):
		return self.__type

	def isConfigured(self):
		return self.__configured

	# Compute keypoint grid for MultiScale Dense SIFT (only once)
	def __computeKeypoint_MSDenseSIFT(self, img_width, img_height):
		# Below, a naive approach to multiscale SIFT (very computationally expensive)
		step = self.__DSstep
		scales = self.__DSscales
		startSize = step
		kpt = []
		x = xrange(step, img_width, step)
		y = xrange(step, img_height, step)
		z = xrange(startSize, step * scales, startSize)
		XX, YY, ZZ = np.meshgrid(x, y, z, indexing='ij')
		XX_f = XX.flatten()
		YY_f = YY.flatten()
		ZZ_f = ZZ.flatten()
		# for i in xrange(D_prm, gray.shape[0], D_prm):
		#	for j in xrange(D_prm, gray.shape[1], D_prm):
		#		for k in xrange(startSize, D_prm * num_scales, startSize):
		#
		for i in xrange(0, len(XX_f)):
			kpt.append(cv2.KeyPoint(float(XX_f[i]), float(YY_f[i]), float(ZZ_f[i])))
	
		return kpt


	# Extracts features of an image given a file path
	def extractFeatures(self, filename):
		#assert(self.__configured)
		kpt = None
		des = None

		ima=cv2.imread(filename)
		gray=cv2.cvtColor(ima,cv2.COLOR_BGR2GRAY)
		
		if self.__type == 'SIFT':
			kpt,des = self.__descriptor.detectAndCompute(gray,None)
			print str(len(kpt))+' '+str(len(des))+' extracted keypoints and descriptors'
		elif self.__type == "DenseSIFT":
			kpt = self.__computeKeypoint_MSDenseSIFT(ima.shape[0], ima.shape[1])
			_, des = self.__descriptor.compute(gray, kpt)
		elif self.__type == 'HOG':
			#winStride = (8,8)
			#padding = (8,8)
			#locations = ((10,20),)
			#downscaled_ima = cv2.resize(ima, (self.__hogWinSize, self.__hogWinSize), interpolation=cv2.INTER_AREA)
			des = self.__descriptor.compute(ima) #,winStride,padding,locations)
		elif self.__type == 'HUEHIST':
			hsv = cv2.cvtColor(ima, cv2.COLOR_BGR2HSV)
			h,s,v = cv2.split(hsv)
			hist,bins = np.histogram(h.ravel(), self.__hueHistogramBins, density=True)
			des = np.array([hist])

		return kpt, des

