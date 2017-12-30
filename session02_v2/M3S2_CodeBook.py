from sklearn import cluster
import cPickle
import time
from M3S2_ImageFeatureExtractor import ImageFeatureExtractor
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# Classifier class that encapsulates a generic classifier
# An instance of this class is expected to be configured first in order
# to impersonate a given classifier (such as KNN). 
# After configuration, it can be train()'ed with a list of descriptors and a 
# list of their labels (np.arrays). 
# We can also predict() a descriptor using the given classifier
class CodeBook(BaseEstimator, TransformerMixin):		#compute the codebook
	#initialize vars
	def __init__(self, k=50, numFeatures=10):	
		self.k = k
		self.numFeatures = numFeatures
		self.codebook = None
		self.descriptor = ImageFeatureExtractor("SIFT")
		self.descriptor.configureSIFT(self.numFeatures)
		print "Called init with numFeatures", numFeatures
	
	def CB (self, descriptors):
		print 'Computing kmeans with '+str(self.k)+' centroids'
		init=time.time()
		self.codebook = cluster.MiniBatchKMeans(n_clusters=self.k, verbose=False, batch_size=self.k * 20,compute_labels=False,reassignment_ratio=10**-4,random_state=42)
		self.codebook.fit(descriptors)
		cPickle.dump(self.codebook, open("codebook.dat", "wb"))
		end=time.time()
		print 'Done in '+str(end-init)+' secs.'
		return self.codebook
	
	def fit(self, X, y=None, **fit_params):
		# X are images filenames
		# fit_params contain SIFT "numFeatures" and KMeans "k"
		# return visual words
		
		print self.numFeatures
	
		self.descriptor.configureSIFT(self.numFeatures)
		
		Train_descriptors = []
		for i in range(len(X)):
			filename=X[i]
			print 'Reading image '+filename
			kpt,des= self.descriptor.extractFeatures(filename)
			Train_descriptors.append(des)
			print str(len(kpt))+' extracted keypoints and descriptors'
		
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
		
		self.descriptor.configureSIFT(self.numFeatures)
		
		Train_descriptors = []
		for i in range(len(X)):
			filename=X[i]
			print 'Reading image '+filename
			kpt,des= self.descriptor.extractFeatures(filename)
			Train_descriptors.append(des)
			print str(len(kpt))+' extracted keypoints and descriptors'
		
		init=time.time()
		visual_words=np.zeros((len(Train_descriptors),self.k),dtype=np.float32)
		for i in xrange(len(Train_descriptors)):
			words = self.codebook.predict(Train_descriptors[i])
			visual_words[i,:]=np.bincount(words,minlength=self.k)
			end=time.time()
		print 'Done in '+str(end-init)+' secs.'
		
		return visual_words