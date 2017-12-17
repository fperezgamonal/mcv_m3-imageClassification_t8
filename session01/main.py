#!/usr/bin/env python2
import time
from M3S1_Pipeline import M3S1_Pipeline

def testSIFT_KNN():
	pipeline = M3S1_Pipeline()
	pipeline.getFeatureExtractor().configureSIFT(60)
	pipeline.getClassifier().configureKNN(8,-1)

	start = time.time()
	pipeline.run()
	end=time.time()

	print 'Done in '+str(end-start)+' secs.'

	pipeline.getEvaluation().printEvaluation()

def testSIFT_Bayes():
	pipeline = M3S1_Pipeline()
	pipeline.getFeatureExtractor().configureSIFT(60)
	pipeline.getClassifier().configureGaussianNBayes()

	start = time.time()
	pipeline.run()
	end=time.time()

	print 'Done in '+str(end-start)+' secs.'

	pipeline.getEvaluation().printEvaluation()

def testSIFT_RandomForest():
	pipeline = M3S1_Pipeline()
	pipeline.getFeatureExtractor().configureSIFT(60)
	pipeline.getClassifier().configureRandomForest()

	start = time.time()
	pipeline.run()
	end=time.time()

	print 'Done in '+str(end-start)+' secs.'

	pipeline.getEvaluation().printEvaluation()

def testHOG_KNN():
	pipeline = M3S1_Pipeline()
	pipeline.getFeatureExtractor().configureHOG(9)
	pipeline.getClassifier().configureKNN(8,-1)

	start = time.time()
	pipeline.run()
	end=time.time()

	print 'Done in '+str(end-start)+' secs.'

	pipeline.getEvaluation().printEvaluation()
	
def testHOG_Bayes():
	pipeline = M3S1_Pipeline()
	pipeline.getFeatureExtractor().configureHOG(9)
	pipeline.getClassifier().configureGaussianNBayes()

	start = time.time()
	pipeline.run()
	end=time.time()

	print 'Done in '+str(end-start)+' secs.'

	pipeline.getEvaluation().printEvaluation()
	
def testHOG_RandomForest():
	pipeline = M3S1_Pipeline()
	pipeline.getFeatureExtractor().configureHOG(9)
	pipeline.getClassifier().configureRandomForest()

	start = time.time()
	pipeline.run()
	end=time.time()

	print 'Done in '+str(end-start)+' secs.'

	pipeline.getEvaluation().printEvaluation()
	
def testHUE_KNN():
	pipeline = M3S1_Pipeline()
	pipeline.getFeatureExtractor().configureHueHistogram(100)
	pipeline.getClassifier().configureKNN(8,-1)

	start = time.time()
	pipeline.run()
	end=time.time()

	print 'Done in '+str(end-start)+' secs.'

	pipeline.getEvaluation().printEvaluation()
	
def testHUE_Bayes():
	pipeline = M3S1_Pipeline()
	pipeline.getFeatureExtractor().configureHueHistogram(100)
	pipeline.getClassifier().configureMultinomialNBayes()

	start = time.time()
	pipeline.run()
	end=time.time()

	print 'Done in '+str(end-start)+' secs.'

	pipeline.getEvaluation().printEvaluation()
	
def testHUE_RForest():
	pipeline = M3S1_Pipeline()
	pipeline.getFeatureExtractor().configureHueHistogram(100)
	pipeline.getClassifier().configureRandomForest()

	start = time.time()
	pipeline.run()
	end=time.time()

	print 'Done in '+str(end-start)+' secs.'

	pipeline.getEvaluation().printEvaluation()
#pipeline.getFeatureExtractor().configureHueHistogram(100)
#pipeline.getClassifier().configureKNN(5,-1)
#pipeline.getClassifier().configureGaussianNBayes()


testHUE_Bayes()