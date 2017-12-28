# TODO:
# FOR NOW: copied code related to features (so we can see in red needed imports,etc.)

import sys
import time
import os.path
import cPickle
import numpy as np
import cv2

# create the SIFT detector object
#SIFTdetector = cv2.SIFT(nfeatures=300)

# Compute training descriptors
def computeTraining_descriptors(D_type, D_prm, train_imgs_filenames, train_labels):
	# Check that we have not computed these features before
	root_folder = os.path.join(os.pardir,"PreComputed_Params/Features/")
	feat_filename = root_folder + "features_train_" + D_type +\
		"_" + str(D_prm) + ".pkl"
	feat_filename = feat_filename
	if os.path.isfile(feat_filename):
		# Load features
		print "Loading features from disk..."
		start = time.time()
		D, train_descriptors = cPickle.load(open(feat_filename, "r"))
		end = time.time()
		print "Features loaded successfully in " + str(end-start) + " secs."
		# return as stacked numpy array 'D' and matrix 'train_descriptors' (for k-means, etc.)
		return D, train_descriptors
	else:
		#  Compute and store descriptors in a python list of numpy arrays
		print "Computing and storing descriptors..."
		start = time.time()
		Train_descriptors = []
		Train_label_per_descriptor = []
		for i in range(len(train_imgs_filenames)):
			filename = train_imgs_filenames[i]
			print 'Reading image '+filename
			ima = cv2.imread(filename)
			kpt, des = compute_imgDescriptor(ima, D_type, D_prm)
			Train_descriptors.append(des)
			Train_label_per_descriptor.append(train_labels[i])
			print str(len(kpt))+' extracted keypoints and descriptors'

		# Transform everything to numpy arrays
		size_descriptors=Train_descriptors[0].shape[1]
		D = np.zeros((np.sum([len(p) for p in Train_descriptors]),size_descriptors),dtype=np.uint8)
		startingpoint = 0
		for i in range(len(Train_descriptors)):
			D[startingpoint:startingpoint+len(Train_descriptors[i])] = Train_descriptors[i]
			startingpoint += len(Train_descriptors[i])

		# Save features to disk to avoid re-computation
		if not os.path.isdir(root_folder):
			# Create folder
			os.mkdir(root_folder)

		cPickle.dump([D, Train_descriptors], open(feat_filename, "wb"))
		end = time.time()
		print "Features computed and stored in " + str(end-start) + " secs."
		return D, Train_descriptors

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
		num_scales = 5
		startSize = D_prm		# equal to the step size
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		try:
			sift = cv2.xfeatures2d.SIFT_create()
		except:
			sift = cv2.SIFT()
		#kpt = [cv2.KeyPoint(x, y, D_prm) for y in range(0, gray.shape[0], D_prm)
		#	   							 for x in range(0, gray.shape[1], D_prm)]
		# Dense SIFT is equal to:

		#des = sift.compute(gray, kpt)
		# Above implementation only contained Dense SIFT w/o multiscale.
		# Below, a naive approach to multiscale SIFT (very computationally expensive)
		kpt = []
		start = time.time()
		for i in xrange(D_prm, gray.shape[0], D_prm):
			for j in xrange(D_prm, gray.shape[1], D_prm):
				for k in xrange(startSize, D_prm * num_scales, startSize):
					kpt.append(cv2.KeyPoint(float(i), float(j), float(k)))

		# Multi-scale SIFT
		kpt, des = sift.compute(gray, kpt)
		end = time.time()
		print "DEBUG: time spent computing 1 images multi-scale Dense SIFT features: " +\
			str(end-start) + " secs."


	elif D_type == "FisherVectors":
		kpt = 0
		des = 0
	else:
		sys.exit("Invalid descriptor type string in function 'compute_imgDescriptor' (features.py)")

	return kpt, des
