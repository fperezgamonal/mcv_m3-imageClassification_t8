#!/usr/bin/env python2
import time
import sys
import os.path
import numpy as np
from M3S2_Pipeline import Pipeline
import cPickle

# NOTE: this script is practically identical to cross_val.py but
# w/o looping through params or calling KFoldCrossValidate...
# (same structure)

# Parameters:
num = 0						# Select scheme (see bottom of script)

# Define cross validation parameters for each scheme and run it
def test_scheme(scheme):
	# 1st classifier: KNN
	if 0 <= scheme <= 2:
		# Load KNN-specific params
		clf_type = 'KNN'
		clf_params = {'n_neighbors': 30}

		# feats: SIFT
		if scheme == 0:
			feat_type = 'SIFT'
			feat_num = 60

		# feats: HueHist
		elif scheme == 1:
			feat_type = 'HueHist'
			feat_num = 104

		# feats: HOG
		else:
			feat_type = 'HOG'
			feat_num = 8

	# 2nd classifier: GaussianN Bayes
	elif 2 < scheme <= 5:
		# Load GaussianN Bayes-specific params
		clf_type = 'GaussNBayes'
		print "No parameter used, only one was available: 'prior' (probability)"

		# feats: SIFT
		if scheme == 3:
			feat_type = 'SIFT'
			feat_num = 60

		# feats: HueHist
		elif scheme == 4:
			feat_type = 'HueHist'
			feat_num = 104

		# feats: HOG
		else:
			feat_type = 'HOG'
			feat_num = 8

	# 3rd classifier: MultinomialN Bayes
	elif 5 < scheme <= 8:
		# Load MultinomialN Bayes-specific params
		clf_type = 'MultinomialNBayes'
		print "No parameter used, three were available: 2 'prior', 1 smoothing factor"

		# feats: SIFT
		if scheme == 6:
			feat_type = 'SIFT'
			feat_num = 60

		# feats: HueHist
		elif scheme == 7:
			feat_type = 'HueHist'
			feat_num = 104

		# feats: HOG
		else:
			feat_type = 'HOG'
			feat_num = 8

	# 4th classifier: Random Forest
	elif 8 < scheme <= 11:
		# Load Random Forest-specific params
		clf_type = 'RForest'
		clf_params = {'n_estimators': 20}

		# feats: SIFT
		if scheme == 9:
			feat_type = 'SIFT'
			feat_num = 60

		# feats: HueHist
		elif scheme == 10:
			feat_type = 'HueHist'
			feat_num = 104

		# feats: HOG
		else:
			feat_type = 'HOG'
			feat_num = 8

	# 5th classifier: SVM
	elif 11 < scheme <= 14:
		# Load SVM-specific params
		clf_type = 'SVM'
		clf_params = {'C': 1,
					   'gamma': 0.056,
					   'kernel': 'rbf',
					   'probability': False}

		# feats: SIFT
		if scheme == 12:
			feat_type = 'SIFT'
			feat_num = 60

		# feats: HueHist
		elif scheme == 13:
			feat_type = 'HueHist'
			feat_num = 104

		# feats: HOG
		else:
			feat_type = 'HOG'
			feat_num = 8

	# Wrong 'num' value, finish execution (to avoid code repetition)
	else:
		sys.exit("Scheme number 'num' out of range 1-14")

	# Common commands (main loop for CV)
	print 'Main run w. feat_type={!s} & clf_type={!s}...'.format(feat_type,clf_type)

	pipeline = Pipeline()

	values = []		# * not returned or saved. Consider deletion

	# SIFT
	if scheme in [0, 3, 6, 9, 12]:
		pipeline.getFeatureExtractor().configureSIFT(feat_num)
	# HueHist
	elif scheme in [1, 4, 7, 10, 13]:
		pipeline.getFeatureExtractor().configureHueHistogram(feat_num)
	# HOG
	elif scheme in [2, 5, 8, 11, 14]:
		pipeline.getFeatureExtractor().configureHOG(feat_num)
	# Default
	else:
		pipeline.getFeatureExtractor().configureSIFT(feat_num)

	# Instantiate the correct classifier type																						 bestParamsValue))
	if clf_type == 'KNN':
		pipeline.getClassifier().configureKNN(clf_params['n_neighbors'], -1)
	elif clf_type == 'GaussNBayes':
		pipeline.getClassifier().configureGaussianNBayes()
	elif clf_type == 'MultinomialNBayes':
		pipeline.getClassifier().configureMultinomialNBayes()
	elif clf_type == 'RForest':
		pipeline.getClassifier().configureRandomForest(clf_params['n_estimators'])
	elif clf_type == 'SVM':
		train_labels = cPickle.load(open('train_labels.dat','r'))
		pipeline.getClassifier().configureSVM(clf_params['C'], clf_params['gamma'],
											  clf_params['kernel'], train_labels)
	else:
		# Load KNN as default (to avoid errors)
		pipeline.getClassifier().configureKNN(clf_params['n_neighbors'], -1)

	start = time.time()
	# In order to avoid re-computing features and/or the model itself, we pass
	# feat and clf info to run() to check if they have already been computed
	pipeline.run(feat_type, feat_num, clf_type, clf_params)
	end = time.time()
	print 'Done in ' + str(end - start) + ' secs.'

	pipeline.getEvaluation().printEvaluation()

if __name__ == "__main__":
	# main of this function
	print "Executing main.py..."

# change 'num' value (top) to run routine number 'num'. Mappings:
#	'num' value			routine
#	  0					SIFT+KNN
#	  1					HueHist+KNN
#     2					HOG+KNN
#	  3					SIFT+GaussBayes
#	  4					HueHist+GaussBayes
#	  5					HOG+GaussBayes
#	  6					SIFT+MultinomialBayes
#	  7					HueHist+MultinomialBayes
# 	  8					HOG+MultinomialBayes
#	  9					SIFT+RandomForests
#	 10    				HueHist+RandomForests
#	 11					HOG+RandomForests
#	 12					SIFT+SVM
#	 13					HueHist+SVM
#	 14					HOG+SVM

# Execute 'num' scheme ('num' is defined at the top)
	test_scheme(num)