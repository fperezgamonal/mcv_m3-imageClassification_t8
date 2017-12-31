import os
import numpy as np
from yael import ynumpy


		
		
class FisherVectors: 
			
	#def LoadDescriptors (image_names, image_descs)
		#We first load all the descriptors
		# list of available images: image_names 
		
		# load the SIFTs for these images
		#image_descs = []
		#for imname in image_names:
		#	desc, meta = ynumpy.siftgeo_read("holidays_100/%s.siftgeo" % imname)
		#	if desc.size == 0: 
		#		desc = np.zeros((0, 128), dtype = 'uint8')
		#	# we drop the meta-information (point coordinates, orientation, etc.)
		#	image_descs.append(desc)
		
	def SampleDescriptors(image_descs, k , n_sample)
		# make a big matrix with all image descriptors
		all_desc = np.vstack(image_descs)

		# choose n_sample descriptors at random
		sample_indices = np.random.choice(all_desc.shape[0], n_sample)
		sample = all_desc[sample_indices]

		# until now sample was in uint8. Convert to float32
		sample = sample.astype('float32')

		# compute mean and covariance matrix for the PCA
		mean = sample.mean(axis = 0)
		sample = sample - mean
		cov = np.dot(sample.T, sample)

		# compute PCA matrix and keep only 64 dimensions
		eigvals, eigvecs = np.linalg.eig(cov)
		perm = eigvals.argsort()                   # sort by increasing eigenvalue
		pca_transform = eigvecs[:, perm[64:128]]   # eigenvectors for the 64 last eigenvalues

		# transform sample with PCA (note that numpy imposes line-vectors,
		# so we right-multiply the vectors)
		sample = np.dot(sample, pca_transform)

		# train GMM
		gmm = ynumpy.gmm_learn(sample, k)
		
		return gmm, pca_transform, mean
		
	def EncodeSift (gmm, image_descs, pca_transform, mean)
		image_fvs = []
		for image_desc in image_descs:
		   # apply the PCA to the image descriptor
		   image_desc = np.dot(image_desc - mean, pca_transform)
		   # compute the Fisher vector, using only the derivative w.r.t mu
		   fv = ynumpy.fisher(gmm, image_desc, include = 'mu')
		   image_fvs.append(fv)

		# make one matrix with all FVs
		image_fvs = np.vstack(image_fvs)

		# normalizations are done on all descriptors at once

		# power-normalization
		image_fvs = np.sign(image_fvs) * np.abs(image_fvs) ** 0.5

		# L2 normalize
		norms = np.sqrt(np.sum(image_fvs ** 2, 1))
		image_fvs /= norms.reshape(-1, 1)

		# handle images with 0 local descriptor (100 = far away from "normal" images)
		image_fvs[np.isnan(image_fvs)] = 100

		return image_fvs
		
	def CompareImages(image_names, image_fvs)
		# get the indices of the query images (the subset of images that end in "0")
		query_imnos = [i for i, name in enumerate(image_names) if name[-1:] == "0"]

		# corresponding descriptors
		query_fvs = image_fvs[query_imnos]

		# get the 8 NNs for all query images in the image_fvs array
		results, distances = ynumpy.knn(query_fvs, image_fvs, nnn = 8)
		
		return query_imnos, results, distances
		
	def mAPPerformance(image_names, query_imnos, results)
		aps = []
		for qimno, qres in zip(query_imnos, results):
			qname = image_names[qimno]
			# collect the positive results in the dataset
			# the positives have the same prefix as the query image
			positive_results = set([i for i, name in enumerate(image_names) if name != qname and name[:4] == qname[:4]])
			#
			# ranks of positives. We skip the result #0, assumed to be the query image
			ranks = [i for i, res in enumerate(qres[1:]) if res in positive_results]
			#
			# accumulate trapezoids with this basis
			recall_step = 1.0 / len(positive_results)
			ap = 0
			for ntp,rank in enumerate(ranks):
			   # ntp = nb of true positives so far
			   # rank = nb of retrieved items so far
			   # y-size on left side of trapezoid:
			   precision_0 = ntp/float(rank) if rank > 0 else 1.0
			   # y-size on right side of trapezoid:
			   precision_1 = (ntp + 1) / float(rank + 1)
			   ap += (precision_1 + precision_0) * recall_step / 2.0
			print "query %s, AP = %.3f" % (qname, ap)
			aps.append(ap)

		print "mean AP = %.3f" % np.mean(aps)
		 
		return np.mean(aps)

	# RUN parameters
	# image_names: type list, contains the name of he images
	# image_descs: type list, contains descriptors of the images
	# mandatory: steps 1 and 2
	# not mandatory: steps 3 and 4
	def run (image_names, image_descs)
		#STEP 1
		print "step 1"
		# Next we sample the descriptors to reduce their dimensionality by PCA and computing a GMM. 
		# For a GMM of size k (let’s set it to 64), we need about 1000*k training descriptors
		# RETUNS: 1) Gausian Mixture Model trained 2) PCA Transform eigenvectors, mean for the PCA
		k = 64
		gmm, pca_transform, mean = SampleDescriptors(image_descs, k, k * 1000)

		#STEP 2
		print "step 2"
		#The training is finished. The next stage is to encode the SIFTs into one vector per image. 
		#We choose to include only the derivatives w.r.t. mu in the FV, which results in a FV of size k * 64.
		# RETUNS: Fisher Vectors
		image_fvs =  EncodeSift (gmm, image_descs, pca_transform, mean)

		#STEP 3
		print "step 3"
		# Now the FV can be used to compare images, so we compute for each Holidays query image the nearest images in the image_fvs matrix.
		# RETUNS: 1) indices of the query images 2)int, classificació 3) float, distancia al cluster
		query_imnos, results, distances = CompareImages(image_names, image_fvs)
		
		#STEP 4
		print "step 4"
		# The mAP performance for this search can be computed as:
		# RETUNS: mAP performance
		mAP = mAPPerformance(image_names, query_imnos, results)
		