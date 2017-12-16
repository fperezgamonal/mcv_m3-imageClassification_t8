#!/usr/bin/env python2
import time
from M3S1_Pipeline import M3S1_Pipeline


# SIFT with 100 features + 5-KNN
scoreslist = []
pipeline = M3S1_Pipeline()

pipeline.getFeatureExtractor().configureSIFT(100)
#pipeline.getFeatureExtractor().configureHueHistogram(100)
pipeline.getClassifier().configureKNN(5,-1)
#pipeline.getClassifier().configureBayes()

start = time.time()
acc, std = pipeline.KFoldCrossValidate()
print("Mean score: {0:.3f} (+/-{1:.3f})".format(acc, std))
end=time.time()

print 'Done in '+str(end-start)+' secs.'	
