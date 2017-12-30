from sklearn import cluster
import cPickle
import time
from M3S2_ImageFeatureExtractor import ImageFeatureExtractor
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import os
import hashlib

# Classifier class that encapsulates a generic classifier
# An instance of this class is expected to be configured first in order
# to impersonate a given classifier (such as KNN). 
# After configuration, it can be train()'ed with a list of descriptors and a 
# list of their labels (np.arrays). 
# We can also predict() a descriptor using the given classifier
class CodeBook(BaseEstimator, TransformerMixin):		#compute the codebook
	#initialize vars
	def __init__(self, k=50, numFeatures=10, descType="SIFT", step=8, scales=5):	
		self.k = k
		self.numFeatures = numFeatures
		self.codebook = None
		self.descType = descType
		self.descriptor = ImageFeatureExtractor(descType)
		self.step = step
		self.scales = scales
		
	def __getFeaturesSaveFilename(self, dataset):
		root_folder = os.path.join(os.pardir, "PreComputed_Params/Features/")
		if not os.path.isdir(root_folder):
			# Create folder
			os.mkdir(root_folder)
			
		feat_filename = root_folder
		feat_filename += "features_" + dataset + "_"
		feat_filename += self.descType
		
		if self.descType == "SIFT":
			feat_filename += "_" + str(self.numFeatures)
		elif self.descType == "DenseSIFT":
			feat_filename += "_st" + str(self.step) + "_sc" + str(self.scales)
		
		feat_filename += ".pkl"
		return feat_filename
	
	def __getModelSaveFilename(self, dataset):
		root_folder = os.path.join(os.pardir, "PreComputed_Params/Codebooks/")
		if not os.path.isdir(root_folder):
			# Create folder
			os.mkdir(root_folder)
			
		model_filename = root_folder
		model_filename += "codebook_" + dataset + "_"
		model_filename += str(self.k)
		model_filename += ".pkl"
		
		return model_filename
		
	def CB (self, descriptors):
		m = hashlib.md5(str(descriptors)).hexdigest()
		model_filename = self.__getModelSaveFilename(m)
		
		print model_filename
		
		if os.path.isfile(model_filename):
			print "Loading codebook saved in ", model_filename
			start = time.time()
			self.codebook = cPickle.load(open(model_filename, "r"))
			# Loaded descriptors (after PCA)
			end = time.time()
			print "Codebook loaded successfully in " + str(end-start) + " secs."
		else:
			print 'Computing kmeans with '+str(self.k)+' centroids'
			init=time.time()
			self.codebook = cluster.MiniBatchKMeans(n_clusters=self.k, verbose=False, batch_size=self.k * 20,compute_labels=False,reassignment_ratio=10**-4,random_state=42)
			self.codebook.fit(descriptors)
			cPickle.dump(self.codebook, open(model_filename, "wb"))
			end=time.time()
			print 'Done in '+str(end-init)+' secs.'
			
		return self.codebook
	
	def __extractFeatures(self, X):
		
		md5 = hashlib.md5(str(X)).hexdigest()
		feat_filename = self.__getFeaturesSaveFilename(md5)
		if os.path.isfile(feat_filename):
			print "Loading features saved in ", feat_filename
			start = time.time()
			Train_descriptors = cPickle.load(open(feat_filename, "r"))
			# Loaded descriptors (after PCA)
			end = time.time()
			print "Features loaded successfully in " + str(end-start) + " secs."
		else:
			if self.descType == "SIFT":
				self.descriptor.configureSIFT(self.numFeatures)
			elif self.descType == "DenseSIFT":
				self.descriptor.configureDenseSIFT(self.step, self.scales)
		
			Train_descriptors = []
			for i in range(len(X)):
				filename=X[i]
				print 'Reading image '+filename
				kpt,des= self.descriptor.extractFeatures(filename)
				Train_descriptors.append(des)
				print str(len(kpt))+' extracted keypoints and descriptors'
			
			cPickle.dump(Train_descriptors, open(feat_filename, "wb"))
			
		return Train_descriptors
	
	def fit(self, X, y=None, **fit_params):
		# X are images filenames
		# fit_params contain SIFT "numFeatures" and KMeans "k"
		# return visual words
		
		Train_descriptors = self.__extractFeatures(X)
			
		# Transform everything to numpy arrays
		size_descriptors=Train_descriptors[0].shape[1]
		D=np.zeros((np.sum([len(p) for p in Train_descriptors]),size_descriptors),dtype=np.uint8)
		startingpoint=0
		for i in range(len(Train_descriptors)):
			D[startingpoint:startingpoint+len(Train_descriptors[i])]=Train_descriptors[i]
			startingpoint+=len(Train_descriptors[i])
			
		self.CB(D)
		
		return self
	
	def transform(self, X):
		# X are images filenames
		print 'Getting Train BoVW representation'
		
		Train_descriptors = self.__extractFeatures(X)
		
		init=time.time()
		visual_words=np.zeros((len(Train_descriptors),self.k),dtype=np.float32)
		for i in xrange(len(Train_descriptors)):
			words = self.codebook.predict(Train_descriptors[i])
			visual_words[i,:]=np.bincount(words,minlength=self.k)
			end=time.time()
		print 'Done in '+str(end-init)+' secs.'
		
		return visual_words