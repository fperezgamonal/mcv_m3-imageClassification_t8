#!/usr/bin/env python2
import time
import numpy as np
from M3S1_Pipeline import M3S1_Pipeline

def testSIFT_KNN():
	pipeline = M3S1_Pipeline()

	values = []
	for feats in range(20,101,20):
		for k in range(3,11):
			pipeline.getFeatureExtractor().configureSIFT(feats)
			#pipeline.getFeatureExtractor().configureHueHistogram(100)
			pipeline.getClassifier().configureKNN(k,-1)
			#pipeline.getClassifier().configureBayes()
			
			start = time.time()
			acc, std = pipeline.KFoldCrossValidate()
			print("Mean score: {0:.3f} (+/-{1:.3f})".format(acc, std))
			end=time.time()
			print 'Done in '+str(end-start)+' secs.'	
			
			values.append([feats, k, acc, std, end-start])
		
	return values

def testSIFT_KNN_ext():
	pipeline = M3S1_Pipeline()

	values = []
	for feats in range(20,101,20):
		for k in range(11,13):
			pipeline.getFeatureExtractor().configureSIFT(feats)
			#pipeline.getFeatureExtractor().configureHueHistogram(100)
			pipeline.getClassifier().configureKNN(k,-1)
			#pipeline.getClassifier().configureBayes()
			
			start = time.time()
			acc, std = pipeline.KFoldCrossValidate()
			print("Mean score: {0:.3f} (+/-{1:.3f})".format(acc, std))
			end=time.time()
			print 'Done in '+str(end-start)+' secs.'	
			
			values.append([feats, k, acc, std, end-start])
		
	return values

# Hue histogram + KNN
def testHueHistogram_KNN():
	pipeline = M3S1_Pipeline()
	print 'Starting cross-validation with Hue + KNN'

	values = []
	for feats in range(8, 256, 32):
		for k in range(3, 11):
			#pipeline.getFeatureExtractor().configureHOG()
			pipeline.getFeatureExtractor().configureHueHistogram(feats)
			pipeline.getClassifier().configureKNN(k, -1)
			# pipeline.getClassifier().configureBayes()

			start = time.time()
			acc, std = pipeline.KFoldCrossValidate()
			print("Mean score: {0:.3f} (+/-{1:.3f})".format(acc, std))
			end = time.time()
			print 'Done in ' + str(end - start) + ' secs.'

			values.append([feats, k, acc, std, end - start])

	return values

# HOG + Random Forests
def testHOG_RForest():
	pipeline = M3S1_Pipeline()
	print 'Starting cross-validation with HOG + RandomForests'

	values = []
	for feats in [1,2,4,8,16]:
		for numEstimators in range(2, 11):
			pipeline.getFeatureExtractor().configureHOG(feats)
			#pipeline.getFeatureExtractor().configureHueHistogram(feats)
			#pipeline.getClassifier().configureKNN(k, -1)
			# pipeline.getClassifier().configureBayes()
			pipeline.getClassifier().configureRandomForest(numEstimators)

			start = time.time()
			acc, std = pipeline.KFoldCrossValidate()
			print("Mean score: {0:.3f} (+/-{1:.3f})".format(acc, std))
			end = time.time()
			print 'Done in ' + str(end - start) + ' secs.'

			values.append([feats, numEstimators, acc, std, end - start])

	return values

# SIFT with feats features + SVM
def testSIFT_SVM():
	# Define test params:

	C_range = np.logspace(-2, 10, 13)
	gamma_range = np.logspace(-9, 3, 13)
	kernel = 1

	pipeline = M3S1_Pipeline()
	print 'Starting cross-validation with SIFT + SVM'

	values = []
	for feats in range(10, 101, 20):
		for C in C_range:
			for gamma in gamma_range:
				pipeline.getFeatureExtractor().configureSIFT(feats)
				# pipeline.getFeatureExtractor().configureHueHistogram(100)
				pipeline.getClassifier().configureSVM(C, gamma, kernel)
				# pipeline.getClassifier().configureBayes()

				start = time.time()
				acc, std = pipeline.KFoldCrossValidate()
				print 'Results for parameters: feats={0:d}, C={0:.3f}, gamma={0:.3f}, kernel={0:d}'.format(feats,C,gamma,kernel)
				("Mean score: {0:.3f} (+/-{1:.3f})".format(acc, std))
				end = time.time()
				print 'Done in ' + str(end - start) + ' secs.'

				values.append([feats, C, gamma, kernel, acc, std, end - start])
	# Return parameters that yielded the maximum accuracy (store the values anyway to plot them)
	maxAccuracy = max(pos[2] for pos in values)
	print 'Maximum accuracy achieved in CV: {0:.3f}'.format(maxAccuracy)
	return values


# HueHist + SVM
def testHue_SVM():
	# Define test params:

	C_range = np.logspace(-2, 10, 13)
	gamma_range = np.logspace(-9, 3, 13)
	kernel = 1

	pipeline = M3S1_Pipeline()
	print 'Starting cross-validation with Hue Hist + SVM'

	values = []
	for feats in range(10, 101, 20):
		for C in C_range:
			for gamma in gamma_range:
				#pipeline.getFeatureExtractor().configureSIFT(feats)
				pipeline.getFeatureExtractor().configureHueHistogram(100)
				pipeline.getClassifier().configureSVM(C, gamma, kernel)
				# pipeline.getClassifier().configureBayes()

				start = time.time()
				acc, std = pipeline.KFoldCrossValidate()
				print 'Results for parameters: feats={0:d}, C={0:.3f}, gamma={0:.3f}, kernel={0:d}'.format(feats,C,gamma,kernel)
				("Mean score: {0:.3f} (+/-{1:.3f})".format(acc, std))
				end = time.time()
				print 'Done in ' + str(end - start) + ' secs.'

				values.append([feats, C, gamma, kernel, acc, std, end - start])
	# Return parameters that yielded the maximum accuracy (store the values anyway to plot them)
	maxAccuracy = max(pos[2] for pos in values)
	print 'Maximum accuracy achieved in CV: {0:.3f}'.format(maxAccuracy)
	return values


# HOG + SVM
def testHOG_SVM():
	# Define test params:

	C_range = np.logspace(-2, 10, 13)
	gamma_range = np.logspace(-9, 3, 13)
	kernel = 1

	pipeline = M3S1_Pipeline()
	print 'Starting cross-validation with HOG + SVM'

	values = []
	for feats in [1,2,4,8,16]:
		for C in C_range:
			for gamma in gamma_range:
				pipeline.getFeatureExtractor().configureHOG(feats)
				#pipeline.getFeatureExtractor().configureSIFT(feats)
				# pipeline.getFeatureExtractor().configureHueHistogram(100)
				pipeline.getClassifier().configureSVM(C, gamma, kernel)
				# pipeline.getClassifier().configureBayes()

				start = time.time()
				acc, std = pipeline.KFoldCrossValidate()
				print 'Results for parameters: feats={0:d}, C={0:.3f}, gamma={0:.3f}, kernel={0:d}'.format(feats,C,gamma,kernel)
				("Mean score: {0:.3f} (+/-{1:.3f})".format(acc, std))
				end = time.time()
				print 'Done in ' + str(end - start) + ' secs.'

				values.append([feats, C, gamma, kernel, acc, std, end - start])
	# Return parameters that yielded the maximum accuracy (store the values anyway to plot them)
	maxAccuracy = max(pos[2] for pos in values)
	print 'Maximum accuracy achieved in CV: {0:.3f}'.format(maxAccuracy)
	return values

def testSIFT_GaussianNBayes():
	pipeline = M3S1_Pipeline()

	values = []
	for feats in range(20,201,20):
		pipeline.getFeatureExtractor().configureSIFT(feats)
		#pipeline.getFeatureExtractor().configureHueHistogram(100)
		pipeline.getClassifier().configureGaussianNBayes()
		#pipeline.getClassifier().configureBayes()
		
		start = time.time()
		acc, std = pipeline.KFoldCrossValidate()
		print("Mean score: {0:.3f} (+/-{1:.3f})".format(acc, std))
		end=time.time()
		print 'Done in '+str(end-start)+' secs.'	
		
		values.append([feats, acc, std, end-start])
		
	return values

def testSIFT_MultinomialNBayes():
	pipeline = M3S1_Pipeline()

	values = []
	for feats in range(20,201,20):
		pipeline.getFeatureExtractor().configureSIFT(feats)
		#pipeline.getFeatureExtractor().configureHueHistogram(100)
		pipeline.getClassifier().configureMultinomialNBayes()
		#pipeline.getClassifier().configureBayes()
		
		start = time.time()
		acc, std = pipeline.KFoldCrossValidate()
		print("Mean score: {0:.3f} (+/-{1:.3f})".format(acc, std))
		end=time.time()
		print 'Done in '+str(end-start)+' secs.'	
		
		values.append([feats, acc, std, end-start])
		
	return values

def testSIFT_RandomForest():
	values = []
	for feats in range(20,201,20):
		pipeline = M3S1_Pipeline()
		pipeline.getFeatureExtractor().configureSIFT(feats)
		#pipeline.getFeatureExtractor().configureHueHistogram(100)
		pipeline.getClassifier().configureRandomForest()
		#pipeline.getClassifier().configureBayes()
		
		start = time.time()
		acc, std = pipeline.KFoldCrossValidate()
		print("Mean score: {0:.3f} (+/-{1:.3f})".format(acc, std))
		end=time.time()
		print 'Done in '+str(end-start)+' secs.'	
		
		values.append([feats, acc, std, end-start])
		
	return values

def testHOG_KNN():
	values = []
	for scale in [1,2,4,8,16]:
		pipeline = M3S1_Pipeline()
		pipeline.getFeatureExtractor().configureHOG(scale)
		#pipeline.getFeatureExtractor().configureHueHistogram(100)
		#pipeline.getClassifier().configureRandomForest()
		pipeline.getClassifier().configureKNN(8,-1)
		
		start = time.time()
		acc, std = pipeline.KFoldCrossValidate()
		print("Mean score: {0:.3f} (+/-{1:.3f})".format(acc, std))
		end=time.time()
		print 'Done in '+str(end-start)+' secs.'	
		
		values.append([scale, acc, std, end-start])
		
	return values


testHOG_KNN()
