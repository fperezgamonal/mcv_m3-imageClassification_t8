#!/usr/bin/env python2
import time
import sys
import numpy as np
from M3S1_Pipeline import M3S1_Pipeline

# Parameters:
num = 0						# Select scheme (see bottom of script)
plotCV = True				# Enable/disable plotting

# Define cross validation parameters for each scheme and run it
def test_scheme(scheme):
	# 1st classifier: KNN
	if 0 <= scheme <= 2:
		# Load KNN-specific params
		clf_type = 'KNN'
		clf_params = {'n_neighbors': [30]}

		# feats: SIFT
		if scheme == 0:
			feat_type = 'SIFT'
			feat_list = [60]		#range(20,101,20)

		# feats: HueHist
		elif scheme == 1:
			feat_type = 'HueHist'
			feat_list = range(8, 256, 32)

		# feats: HOG
		else:
			feat_type = 'HOG'
			feat_list = [1, 2, 4, 8, 16]

	# 2nd classifier: GaussianN Bayes
	elif 2 < scheme <= 5:
		# Load GaussianN Bayes-specific params
		clf_type = 'GaussNBayes'
		print "No parameter used, only one was available: 'prior' (probability)"

		# feats: SIFT
		if scheme == 3:
			feat_type = 'SIFT'
			feat_list = range(20, 101, 20)

		# feats: HueHist
		elif scheme == 4:
			feat_type = 'HueHist'
			feat_list = range(8, 256, 32)

		# feats: HOG
		else:
			feat_type = 'HOG'
			feat_list = [1, 2, 4, 8, 16]

	# 3rd classifier: MultinomialN Bayes
	elif 5 < scheme <= 8:
		# Load MultinomialN Bayes-specific params
		clf_type = 'MultinomialNBayes'
		print "No parameter used, three were available: 2 'prior', 1 smoothing factor"

		# feats: SIFT
		if scheme == 6:
			feat_type = 'SIFT'
			feat_list = range(20, 101, 20)

		# feats: HueHist
		elif scheme == 7:
			feat_type = 'HueHist'
			feat_list = range(8, 256, 32)

		# feats: HOG
		else:
			feat_type = 'HOG'
			feat_list = [1, 2, 4, 8, 16]

	# 4th classifier: Random Forest
	elif 8 < scheme <= 11:
		# Load Random Forest-specific params
		clf_type = 'RForest'
		clf_params = {'n_estimators': range(1, 100)}

		# feats: SIFT
		if scheme == 9:
			feat_type = 'SIFT'
			feat_list = range(20, 101, 20)

		# feats: HueHist
		elif scheme == 10:
			feat_type = 'HueHist'
			feat_list = range(8, 256, 32)

		# feats: HOG
		else:
			feat_type = 'HOG'
			feat_list = [1, 2, 4, 8, 16]

	# 5th classifier: SVM
	elif 11 < scheme <= 14:
		# Load SVM-specific params
		clf_type = 'SVM'
		clf_params = {'C': np.logspace(-2, 10, 13),
					   'gamma': np.logspace(-9, 3, 13),
					   'kernel': ('linear', 'rbf'),
					   'probability': False}

		# feats: SIFT
		if scheme == 12:
			feat_type = 'SIFT'
			feat_list = range(20, 101, 20)

		# feats: HueHist
		elif scheme == 13:
			feat_type = 'HueHist'
			feat_list = range(8, 256, 32)

		# feats: HOG
		else:
			feat_type = 'HOG'
			feat_list = [1, 2, 4, 8, 16]

	# Wrong 'num' value, finish execution (to avoid code repetition)
	else:
		sys.exit("Scheme number 'num' out of range 1-14")

	# Common commands (main loop for CV)
	print 'CV w. feat_type={!s} & clf_type={!s}...'.format(feat_type,clf_type)

	pipeline = M3S1_Pipeline()

	values = []		# * not returned or saved. Consider deletion
	for feats in feat_list:
		# SIFT
		if scheme in [0, 3, 6, 9, 12]:
			pipeline.getFeatureExtractor().configureSIFT(feats)
		# HueHist
		elif scheme in [1, 4, 7, 10, 13]:
			pipeline.getFeatureExtractor().configureHueHistogram(feats)
		# HOG
		elif scheme in [2, 5, 8, 11, 14]:
			pipeline.getFeatureExtractor().configureHOG(feats)
		# Default
		else:
			pipeline.getFeatureExtractor().configureSIFT(feats)

		start = time.time()
		clfGrid = pipeline.KFoldCrossValidate(clf_params, clf_type, feat_type, feats)
		bestParamsValue = clfGrid.best_params_
		bestParamsScore = clfGrid.best_score_
		print("CV Best score for clf 'KNN' with feats 'HOG': {0:.3f} with params %s)".format(bestParamsScore,
																							 bestParamsValue))
		end = time.time()
		print 'Done in ' + str(end - start) + ' secs.'

		# see (*) above referring 'values'
		values.append([feats, bestParamsValue, bestParamsScore, end - start])

	# Call plot function if enabled
	print 'The CV Grid has been successfully computed...'
	if plotCV:
		plotCV_results(CV_grid=clfGrid, feat_type=feat_type, clf_type=clf_type)
	else:
		print 'Cross Validation operations completed. Exiting function...'

	# Return 'clfGrid' in case we need it afterwards
	return clfGrid

# Define a function to plot the cross-validation results
def plotCV_results(CV_grid, feat_type, clf_type):
	# 1st: print summary to screen
	print "Printing best CV results..."
	print "Feat. type: {!s}; Clf. type: {!s}".format(feat_type,clf_type)

	print("Best parameters set found on development set:")
	print()
	print(CV_grid.best_params_)
	print()
	print("Grid scores on development set:")
	print()
	means = CV_grid.cv_results_['mean_test_score']
	stds = CV_grid.cv_results_['std_test_score']
	for mean, std, params in zip(means, stds, CV_grid.cv_results_['params']):
		print("%0.3f (+/-%0.03f) for %r"
			  % (mean, std * 2, params))
	print()

	# 2nd: generate graph with the results
	print "Plotting CV results...(to be implemented)"

if __name__ == "__main__":
	# main of this function
	print "Executing cross_val.py..."

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