#!/usr/bin/env python2
import time
from M3S1_Pipeline import M3S1_Pipeline


# SIFT with 100 features + 5-KNN
pipeline = M3S1_Pipeline()

pipeline.getFeatureExtractor().configureSIFT(50)
#pipeline.getFeatureExtractor().configureHueHistogram(100)
pipeline.getClassifier().configureKNN(5,-1)

start = time.time()
pipeline.run()
end=time.time()

print 'Done in '+str(end-start)+' secs.'

pipeline.getEvaluation().printEvaluation()
