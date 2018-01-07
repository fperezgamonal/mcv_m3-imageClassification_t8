from sklearn import cluster
import cPickle
import time
from M3S2_ImageFeatureExtractor import ImageFeatureExtractor
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import os
import hashlib
import math

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
		
		if self.descType == "SIFT" or self.descType == "SpatialPyramids":
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
		print("Computing codebook...")
		m = hashlib.md5(str(descriptors[0:10])).hexdigest()
		print("Hash caluclated")
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
			if self.descType == "SIFT" or self.descType == "SpatialPyramids":
				self.descriptor.configureSIFT(self.numFeatures)
			elif self.descType == "DenseSIFT":
				self.descriptor.configureDenseSIFT(self.step, self.scales)
		
			Train_descriptors = []
			for i in range(len(X)):
				filename=X[i]
				print 'Reading image '+filename
				kpt,des= self.descriptor.extractFeatures(filename)
				kpts = [k.pt for k in kpt]
				Train_descriptors.append([kpts,des])
				print str(len(kpt))+' extracted keypoints and descriptors'
			
			cPickle.dump(Train_descriptors, open(feat_filename, "wb"))
			
		return Train_descriptors
	
	def fit(self, X, y=None, **fit_params):
		# X are images filenames
		# fit_params contain SIFT "numFeatures" and KMeans "k"
		# return visual words
		
		Train_descriptors = self.__extractFeatures(X)
			
		# Transform everything to numpy arrays
		kpt0, des0 = Train_descriptors[0]
		size_descriptors=des0.shape[1]
		D=np.zeros((np.sum([len(p) for k,p in Train_descriptors]),size_descriptors),dtype=np.uint8)
		startingpoint=0
		for i in range(len(Train_descriptors)):
			kpt, des = Train_descriptors[i]
			D[startingpoint:startingpoint+len(des)]=des
			startingpoint+=len(des)
			
		self.CB(D)
		
		return self
	
	def transform(self, X):
		# X are images filenames
		print 'Getting Train BoVW representation'
		np.set_printoptions(threshold=np.inf)
		
		Train_descriptors = self.__extractFeatures(X)
		
		init=time.time()
				
		if self.descType == "SIFT" or self.descType == "DenseSIFT":
			visual_words=np.zeros((len(Train_descriptors),self.k),dtype=np.float32)
			for i in xrange(len(Train_descriptors)):
				_, des = Train_descriptors[i]
				words = self.codebook.predict(des)
				visual_words[i,:]=np.bincount(words,minlength=self.k)
				end=time.time()
				
		elif self.descType == "SpatialPyramids":
			visual_words=np.zeros((len(Train_descriptors),self.k*21),dtype=np.float32)
			
			
			step = 256//4
			for i in xrange(len(Train_descriptors)):
				layer = []
				for j in range(16):
					layer.append([])
				
				kpts,dess = Train_descriptors[i]
				words = self.codebook.predict(dess)
				
				# let's divide the image in 16 tiles (4x4)
				# for each keypoint, we get the tile number
				# and create a list for each tile containing
				# the keypoints and descriptors in it
				for kpt, des in zip(kpts,words):
					pos = int(math.floor(((kpt[1]//step)*4) + (kpt[0]//step)))
					layer[pos].append(des)
			
				# for each tile, we compute the histogram
				for j in range(16):
					if len(layer[j]) > 0:
						hist = np.bincount(layer[j],minlength=self.k)#[0:-1]
						visual_words[i,((5+j)*self.k):(((6+j)*self.k))]=hist/2.0
						
						if   j in [0,1,4,5]:
							visual_words[i,(1*self.k):(2*self.k)] = np.add(visual_words[i,(1*self.k):(2*self.k)], hist/4.0)
						elif j in [2,3,6,7]:
							visual_words[i,(2*self.k):(3*self.k)] = np.add(visual_words[i,(2*self.k):(3*self.k)], hist/4.0)
						elif j in [8,9,12,13]:
							visual_words[i,(3*self.k):(4*self.k)] = np.add(visual_words[i,(3*self.k):(4*self.k)], hist/4.0)
						elif j in [10,11,14,15]:
							visual_words[i,(4*self.k):(5*self.k)] = np.add(visual_words[i,(4*self.k):(5*self.k)], hist/4.0)
							
						visual_words[i,0:self.k] = np.add(visual_words[i,0:self.k], hist/4.0)
				
		end=time.time()
		
		print 'Done in '+str(end-init)+' secs.'
		
		print visual_words.shape
		return visual_words
	
	@staticmethod
	def histogramIntersectionKernel(X, Y):
		kernel = np.zeros((X.shape[0], Y.shape[0]))

		for d in range(X.shape[1]):
			column_1 = X[:, d].reshape(-1, 1)
			column_2 = Y[:, d].reshape(-1, 1)
			kernel += np.minimum(column_1, column_2.T)

		return kernel

	@staticmethod
	def pyramidMatchKernel(X, Y):
		k = X.shape[1]/21
		
		X_2 = X[:,0:k]
		Y_2 = Y[:,0:k]
		
		X_1 = X[:,k:5*k]
		Y_1 = Y[:,k:5*k]
		
		X_0 = X[:,5*k:X.shape[1]]
		Y_0 = Y[:,5*k:Y.shape[1]]
		
		I_0 = CodeBook.histogramIntersectionKernel(X_0, Y_0)
		I_1 = CodeBook.histogramIntersectionKernel(X_1, Y_1)
		I_2 = CodeBook.histogramIntersectionKernel(X_2, Y_2)
		
		return I_0 + ((I_1 - I_0)/2.0) + ((I_2 - I_1)/4.0)