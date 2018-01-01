# BoVW.py (SVM)
# TODO:
# Functions to include:
#   - Create BoVW (mini-batch k-means clustering==> codebook, store it)
#   - Predict train(based on the codebook, obtain Visual Words)
#   - Predict test (" " " " " " " on test)
#
# Parameters necessary for later integration with CV
#   - We only need the descriptors 'D' (Dense SIFT or whatever)
#
# NOTES:
#   - Encode codebook.dat in such a way that we only recompute it if:
#       * D is difference (e.g.: different number of dense SIFT vectors per img)
#   - Create a folder called 'codebooks' at the root path of the repo
#   - We work with both D (re-shaped numpy array) and train_descriptors (for testing)
#     which is not re-shaped yet (get them both as input params, properly encoded)

# FOR NOW: copied code related to features (so we can see in red needed imports,etc.)

import cPickle
import os.path
import time
import numpy as np
import cv2
from sklearn import cluster
from sklearn.decomposition import PCA
from features import compute_imgDescriptor

# compute the codebook
def computeCodebook(numClusters, descriptors, D_type, D_prm, PCA_on):       # to start num_clusters was 512
	# Check if we have already computed this codebook
	root_folder = os.path.join(os.pardir, "PreComputed_Params/Codebooks/")
	if D_type == 'Dense_SIFT':
		num_scales = 5
		D_prm = np.append(D_prm, num_scales)

		if PCA_on:
			CB_filename = root_folder + "codebook_k" + str(numClusters) + "-" \
					  	+ D_type + "_" + str(D_prm[0]) + "_PCAon_" +\
					  	str(PCA_on) + ".dat"
		else:
			CB_filename = root_folder + "codebook_k" + str(numClusters) + "-"\
					  	+ D_type + "_" + str(D_prm[0]) + ".dat"
	else:
		if PCA_on:
			CB_filename = root_folder + "codebook_k" + str(numClusters) + "-" \
						  + D_type + "_" + str(D_prm[0]) + "_PCAon_" + \
						  str(PCA_on) + ".dat"
		else:
			CB_filename = root_folder + "codebook_k" + str(numClusters) + "-" \
						  + D_type + "_" + str(D_prm[0]) + ".dat"
	#CB_filename = os.path.join(os.pardir, CB_filename)

	if os.path.isfile(CB_filename):
		# Load it instead of computing it
		print "Loading codebook from disk..."
		codebook = cPickle.load(open(CB_filename, "r"))
		print "Coadebook successfully loaded"
	else:
		# We have to compute it
		print "Computing kmeans with " + str(numClusters) + " centroids"
		initCB = time.time()

		codebook = cluster.MiniBatchKMeans(n_clusters=numClusters, verbose=False,
										   batch_size=numClusters * 20,
										   compute_labels=False,
										   reassignment_ratio=10**-4,
										   random_state=42)
		codebook.fit(descriptors)

		# Save codebook for future usage (avoiding re-computation)
		if not os.path.isdir(root_folder):
			# Create folder
			os.mkdir(root_folder)
		cPickle.dump(codebook, open(CB_filename, "wb"))

		endCB=time.time()
		print "Codebook computed. Done in " + str(endCB-initCB) + " secs."
	# Return codebook
	return codebook

# get train visual word encoding
def getBoVW_train(codebook, numClusters, train_descriptors):
	print "Getting Train BoVW representation"
	initBT = time.time()
	visual_words = np.zeros((len(train_descriptors),numClusters),dtype=np.float32)
	for i in xrange(len(train_descriptors)):
		words = codebook.predict(train_descriptors[i])
		visual_words[i,:] = np.bincount(words,minlength=numClusters)
	endBT = time.time()
	print "BoVW train computed. Done in " + str(endBT-initBT) + " secs."
	return visual_words


# get all the test data
def getBoVW_test(codebook, numClusters, test_imgs_filenames, D_type, D_prm, kpt_dense, PCA_on, pca_train, sclr_train):
	print "Getting Test BoVW representation"
	initBTs=time.time()
	visual_words_test=np.zeros((len(test_imgs_filenames),numClusters),dtype=np.float32)
	if D_type == 'Dense_SIFT':
		D_prm = kpt_dense

	for i in range(len(test_imgs_filenames)):
		filename=test_imgs_filenames[i]
		print "Reading image " + filename
		ima=cv2.imread(filename)
		_, des = compute_imgDescriptor(ima, D_type, D_prm)

		if PCA_on:
			# Apply PCA computed in training
			# Scale the descriptor first
			des = sclr_train.transform(des)
			des = pca_train.transform(des)

		words=codebook.predict(des)
		visual_words_test[i,:]=np.bincount(words,minlength=numClusters)
	endBTs=time.time()
	print "BoVW test computed. Done in " + str(endBTs-initBTs) + " secs."
	return visual_words_test