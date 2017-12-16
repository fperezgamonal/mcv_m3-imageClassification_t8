#!/usr/bin/env python2
import time
from M3S1_Pipeline import M3S1_Pipeline


# SIFT with 100 features + 5-KNN
scoreslist = []
for i in range(5,10):
	pipeline = M3S1_Pipeline()

	pipeline.getFeatureExtractor().configureSIFT(100)
	#pipeline.getFeatureExtractor().configureHueHistogram(100)
	pipeline.getClassifier().configureKNN(5,-1)
	#pipeline.getClassifier().configureBayes()

	start = time.time()
	scoreslist.append(pipeline.KFoldCrossValidate())
	end=time.time()

	print 'Done in '+str(end-start)+' secs.'	
	

for scores in scoreslist:
	print("Mean score: {0:.3f} (+/-{1:.3f})".format(scores.mean(), scores.std()))
	
	
	

