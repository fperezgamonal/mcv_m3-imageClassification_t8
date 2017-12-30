# Based on main.py(session02) and cross_val.py from session01 but simplified
# (as we only use SVM as classifier)
import cPickle
import numpy as np
import time
import os.path
from features import computeTraining_descriptors
from BoVW import computeCodebook, getBoVW_train
from classifier import KFoldCrossValidation
from evaluation import CVPlot_SVM

# Define general variables and select the scheme to run with 'num':
num_scheme = 1
# Initialise variables to 'scheme'-specific values
if num_scheme == 0:
	D_type = "SIFT"
	D_param = np.array([50, 100, 150])
	print "With feature type: " + D_type
elif num_scheme == 1:
	D_type = "Dense_SIFT"
	# step already tested 10 (super-slow)
	D_param = np.array([16, 32, 64]) # step in pixels between two SIFT descriptors
	print "With feature type: " + D_type
elif num_scheme == 2:
	D_type = "FisherVectors"
	D_param = np.array([4, 8, 12])  # the number of GMMs used
	print "With feature type: " + D_type

# Codebook
n_clusters = np.array([128, 256, 512])
# SVM
C_range = 10. ** np.arange(-3, 8)
gamma_range = 10. ** np.arange(-5, 4)
#clf_prms = {'kernel':('rbf', 'precomputed'), 'C':C_range, 'gamma':gamma_range}
clf_prms = {'C':C_range, 'gamma':gamma_range}
num_folds = 5
# C = 1
# gamma = 0.002
# kernel = 'rbf'

# Evaluation
plotGraph = True
PCA_on = False
if PCA_on:
	n_cols = 64
else:
	n_cols = 128

def run_scheme(scheme, descriptor_type, descriptor_param,
			   num_clusters, clf_params, n_folds, plotGraphs, PCA, num_cols):
	print "Running cross-validation scheme number= ..." + str(scheme)


	start = time.time()

	# 1) Read the train and test files (DO ONCE)
	train_images_filenames = cPickle.load(open('train_images_filenames.dat','r'))
	test_images_filenames = cPickle.load(open('test_images_filenames.dat','r'))
	train_labels = cPickle.load(open('train_labels.dat','r'))
	test_labels = cPickle.load(open('test_labels.dat','r'))
	print 'Loaded '+str(len(train_images_filenames))+' training images filenames\
	 with classes ',set(train_labels)
	print 'Loaded '+str(len(test_images_filenames))+' testing images filenames\
	 with classes ',set(test_labels)

	# Create meshgrid of feature parameters and num_clusters to iterate through:
	NC, DP = np.meshgrid(num_clusters, descriptor_param, indexing='ij')
	NC_f = NC.flatten()
	DP_f = DP.flatten()
	for params in xrange(0, len(NC_f)):
		# Get current params:
		descriptor_param = DP_f[params]
		num_clusters = NC_f[params]
		print "Running cross-validation with:" \
			  " feature value: n_clusters=" + str(num_clusters) +\
			", D_param=" + str(descriptor_param)
		print "We iterate through clf_params for each pair of" \
			  " params (n_clust, D_prm)"
		# 2) Extract train features (DO ONE PER COMBINATION OF FEAT_VALUES)
		descriptors_np, Train_descriptors, kpt_dense = \
		computeTraining_descriptors(descriptor_type,
									descriptor_param,
									train_images_filenames, train_labels,
									PCA, num_cols)

		# 3) Reduce number of features by PCA (reducing m=128 cols)


		# descriptors_np = reduceDimensionality_PCA(descriptors_np, num_cols)
		# ############ CHECK TRAIN_DESCRIPTORS DIMENSION, TO PROPERLY APPLY PCA
		# for mtx in Train_descriptors:
		# 	# Apply PCA to reduce dimension
		# 	Train_descriptors[mtx] = reduceDimensionality_PCA(Train_descriptors, num_cols)

		# 4) Compute codebook (DO ONCE PER DIFFERENT NUM_CLUSTERS VALUE)
		codebook = computeCodebook(num_clusters, descriptors_np, descriptor_type,
							   descriptor_param, PCA)
		# 5) Get training BoVW (DO ONCE PER DIFFERENT COMBINATIONS OF FEAT AND CLUSTER VALUES)
		train_VW = getBoVW_train(codebook, num_clusters, Train_descriptors)

		# 6) Cross-validate SVM
		train_scaler, CVGrid = KFoldCrossValidation(train_VW, train_labels, n_folds, clf_params)

		# 7) Print results and generate graphs for cross-validation
		# print "Cross Validation: summary of results..."
		# print ""
		# printCV_resultSummary(CVGrid, descriptor_type)
		print "Cross Validation: plotting accuracy vs (C, gamma)..."
		CVPlot_SVM(CVGrid, clf_prms)

		# Save CV_Grid
		root_folder = os.path.join(os.pardir, "PreComputed_Params/CV-results/")
		grid_filename = root_folder + "D_type=" + str(descriptor_type) +\
			"_Dprm=" + str(descriptor_param) + "_nclusters=" +\
			str(num_clusters) + "_PCAon=" + str(PCA) + ".pkl"

		if not os.path.isdir(root_folder):
			# Create folder
			os.mkdir(root_folder)

		cPickle.dump([CVGrid], open(grid_filename, "wb"))
		# 8) Once this script has finished execution, re-train the best
		#  classifier with the whole training set (using main.py w.
		#  the updated params).
		end=time.time()
		print 'CV done in '+str(end-start)+' secs.'

if __name__ == "__main__":
	# main of this function
	print "Executing session02/cross_val.py..."

# change 'num' value (top) to run routine number 'num'. Mappings:
#	'num' value			routine
#	  0					(BoVW) SIFT + SVM
#	  1					(BoVW) Dense SIFT + SVM
#     2					(BoVW) Fisher Vectors + SVM
#	  N                 (BoVW) routineXX

# Execute 'num' scheme ('num' is defined at the top)
	run_scheme(num_scheme, D_type, D_param, n_clusters, clf_prms, num_folds, plotGraph, PCA_on, n_cols)