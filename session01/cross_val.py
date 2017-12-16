#!/usr/bin/env python2
import time
from M3S1_Pipeline import M3S1_Pipeline


# SIFT with 100 features + 5-KNN
def testSIFT_KNN():
	pipeline = M3S1_Pipeline()

	values = []
	for feats in range(10,101,20):
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
	for feats in range(10,101,20):
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

print testSIFT_KNN_ext()

