# TODO:
# FOR NOW: copied code related to features (so we can see in red needed imports,etc.)

import sys
import time
import os.path
import cPickle
import numpy as np
import cv2
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Compute keypoint grid for MultiScale Dense SIFT (only once)
def computeKeypoint_MSDenseSIFT(D_prm, img_width, img_height):
	# Below, a naive approach to multiscale SIFT (very computationally expensive)
	# D_prm[0]: step
	# D_prm[1]: num_scales
	startSize = D_prm[0]
	kpt = []
	x = xrange(D_prm[0], img_width, D_prm[0])
	y = xrange(D_prm[0], img_height, D_prm[0])
	z = xrange(startSize, D_prm[0] * D_prm[1], startSize)
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

# Compute training descriptors
def computeTraining_descriptors(D_type, D_prm, train_imgs_filenames, train_labels, PCA_on, num_cols):
	# 1) Define root folder and filename to store features
	root_folder = os.path.join(os.pardir, "PreComputed_Params/Features/")
	if PCA_on:
		if D_type == "Dense_SIFT":
			num_scales = 5
			D_prm = np.append(D_prm, num_scales)
			feat_filename = root_folder + "features_train_" + D_type + \
							"_" + str(D_prm[0]) + "_" + str(num_scales) +\
							"_PCA=" + str(PCA_on) + ".pkl"
		else:
			feat_filename = root_folder + "features_train_" + D_type + \
							"_" + str(D_prm[0]) + "_PCA=" + str(PCA_on) +\
							".pkl"
	else:	# No PCA
		if D_type == "Dense_SIFT":
			num_scales = 5
			D_prm = np.append(D_prm, num_scales)

			feat_filename = root_folder + "features_train_" + D_type + \
							"_" + str(D_prm[0]) + "_" + str(num_scales) + ".pkl"
		else:
			feat_filename = root_folder + "features_train_" + D_type + \
							"_" + str(D_prm[0]) + ".pkl"

	# 2) Check if we can load features (otherwise, compute them)
	if os.path.isfile(feat_filename):
		# Load features (PCA)
		print "Loading features from disk..."
		start = time.time()
		if PCA_on:
			D, Train_descriptors, pca, sclr_Dnp = cPickle.load(open(feat_filename, "r"))
			end = time.time()
			print "Features loaded successfully in " + str(end - start) + " secs."
			if D_type == "Dense_SIFT":
				tmp_im = cv2.imread(train_imgs_filenames[0])
				kpt = computeKeypoint_MSDenseSIFT(D_prm, tmp_im.shape[0],
												  tmp_im.shape[1])
				D_prm = kpt
			return D, Train_descriptors, D_prm, pca, sclr_Dnp

		else:
			# Load features (no PCA)
			pca = []
			sclr_Dnp = []
			D, Train_descriptors = cPickle.load(open(feat_filename, "r"))
			end = time.time()
			print "Features loaded successfully in " + str(end - start) + " secs."
			if D_type == "Dense_SIFT":
				tmp_im = cv2.imread(train_imgs_filenames[0])
				kpt = computeKeypoint_MSDenseSIFT(D_prm, tmp_im.shape[0],
												  tmp_im.shape[1])
				D_prm = kpt

			return D, Train_descriptors, D_prm, pca, sclr_Dnp

	else:
		#  Compute and store descriptors in a python list of numpy arrays
		print "Computing and storing descriptors..."
		start = time.time()
		if D_type == "Dense_SIFT":
			tmp_im = cv2.imread(train_imgs_filenames[0])
			kpt = computeKeypoint_MSDenseSIFT(D_prm, tmp_im.shape[0],
											  tmp_im.shape[1])
			D_prm = kpt

		Train_descriptors = []
		Train_label_per_descriptor = []
		for i in range(len(train_imgs_filenames)):
			filename = train_imgs_filenames[i]
			print 'Reading image ' + filename
			ima = cv2.imread(filename)
			kpt, des = compute_imgDescriptor(ima, D_type, D_prm)
			Train_descriptors.append(des)
			Train_label_per_descriptor.append(train_labels[i])
			print str(len(kpt)) + ' extracted keypoints and descriptors'

		# Transform everything to numpy arrays
		size_descriptors = Train_descriptors[0].shape[1]
		D = np.zeros((np.sum([len(p) for p in Train_descriptors]), size_descriptors), dtype=np.uint8)
		startingpoint = 0
		for i in range(len(Train_descriptors)):
			D[startingpoint:startingpoint + len(Train_descriptors[i])] = Train_descriptors[i]
			startingpoint += len(Train_descriptors[i])

		# Check if the root folder exists. Otherwise, create it
		if not os.path.isdir(root_folder):
			os.mkdir(root_folder)

		if PCA_on:
			# Need to scale and apply PCA to descriptors before storing
			# the descriptors and the 'pca' matrix (needed for test)

			# 1) Normalise descriptors
			sclr_Dnp = StandardScaler().fit(D)
			D = sclr_Dnp.transform(D)
			# 2) Apply PCA
			D, pca = reduceDimensionality_PCA(D, num_cols)
			# 3) Apply these transformations (normalisation + PCA)
			#  to the descriptors in "list of arrays" format
			i = 0
			for mtx in Train_descriptors:
				# Apply PCA to reduce dimension
				temp = sclr_Dnp.transform(mtx)
				Train_descriptors[i] = pca.transform(temp)
				i += 1
			# 4) Store descriptors + pca matrix
			cPickle.dump([D, Train_descriptors, pca, sclr_Dnp], open(feat_filename, "wb"))
			end = time.time()
			print "Features computed and stored in " + str(end - start) + " secs."
		else:
			# Directly save computed descriptors to disk (w/o PCA)
			cPickle.dump([D, Train_descriptors], open(feat_filename, "wb"))
			end = time.time()
			print "Features computed and stored in " + str(end - start) + " secs."
			pca = []
			sclr_Dnp = []
			return D, Train_descriptors, D_prm, pca, sclr_Dnp

# Compute a descriptor image per image (used in test)
def compute_imgDescriptor(img, D_type, D_prm):
	if D_type == "SIFT":
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		try:
			sift = cv2.xfeatures2d.SIFT_create(nfeatures=D_prm)
		except:
			sift = cv2.SIFT(nfeatures=D_prm)

		kpt, des = sift.detectAndCompute(gray, None)


	# NOTE: for now we are trying a naive implementation by defining and iterating the grid
	# ourselves (opencv 3 does not support Dense SIFT out of the box. VLFeat is faster (in
	# case this is very slow)
	# We select 5 scales as suggested in this paper (8, 16, 24, 32 and 40): https://goo.gl/srof24

	elif D_type == "Dense_SIFT":		# D_prm is the step between the grid points
		# pass the grid directly (we only need to compute it ONCE!!)
		# num_scales = 5
		# startSize = D_prm		# equal to the step size
		kpt = D_prm
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		try:
			sift = cv2.xfeatures2d.SIFT_create()
		except:
			sift = cv2.SIFT()

		# Multi-scale SIFT
		_, des = sift.compute(gray, kpt)


	elif D_type == "FisherVectors":
		kpt = 0
		des = 0
	else:
		sys.exit("Invalid descriptor type string in function 'compute_imgDescriptor' (features.py)")

	return kpt, des

def reduceDimensionality_PCA(descriptors, SIFT_cols):
	# Reduce the 128 original cols in the SIFT/DenseSIFT
	#  to 'SIFT_cols' columns.
	if descriptors.shape[1] < SIFT_cols:
		print "Descriptor has fewer columns that target number: dimensionality not" \
			  "reduced!"
		return descriptors # do not alter the vector
	else:
		pca = PCA(n_components=SIFT_cols)
		descriptors_out = pca.fit_transform(descriptors)

		return descriptors_out, pca


# Function to compute Spatial Pyramid
# Workflow:
# img => compute descriptors and KEYPOINTS ==>
# distribute descriptors among regions (w.keypoints location)==>
# get one BoVW for each partition==>
# Normalise and perform weighted addition of hists
# (to combine them)
# * With 3 levels==> weights: 1/4 for level 0 and 1, 1/2 level 2
# * 1, 4 and 16 histograms, respectively
#
#	- Notes:
#		* It is better to use the same vocabulary for all
#		*
# ==> train SVM ==> test
# Papers detailing spatial pyramids: https://goo.gl/unM87z &
#									 https://goo.gl/Wc2xea

def compute_spatialPyramids_trainingDescriptors(D_type, D_prm, train_imgs_filenames, train_labels):
	return 0