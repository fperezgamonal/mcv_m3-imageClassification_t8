#!/usr/bin/env python2
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn import cluster

# Classifier class that encapsulates a generic classifier
# An instance of this class is expected to be configured first in order
# to impersonate a given classifier (such as KNN). 
# After configuration, it can be train()'ed with a list of descriptors and a 
# list of their labels (np.arrays). 
# We can also predict() a descriptor using the given classifier
class Classifier:
	
	__slots__=['__configured', '__knn']

	#initialize vars
	def __init__(self):
		self.__configured = False
		self.__classifier = None

	def getType(self):
		return self.__type

	def isConfigured(self):
		return self.__configured

    # each different type of classifier implemented in this class
    # must have its own configuration method in order to instantiate
    # the classifier with a set of parameters
    # configuration of the classifier is mandatory (see self.__configured usage)
	def configureKNN(self, numNeighbors, numJobs):
		self.__classifier = KNeighborsClassifier(n_neighbors=numNeighbors, n_jobs=numJobs)   
		self.__configured = True

	def configureGaussianNBayes(self):
		self.__classifier = GaussianNB()
		self.__configured = True
	
	def configureMultinomialNBayes(self, alpha = 1.0):
		self.__classifier = MultinomialNB()
		self.__configured = True
	
	def configureRandomForest(self, numEstimators):
		self.__classifier = RandomForestClassifier(n_estimators=numEstimators)
		self.__configured = True

	def configureSVM(self, C, gamma, kernel, train_labels):
		print 'Training the SVM classifier...'
		# Create and configure SVM
		stdSlr = StandardScaler().fit(visual_words)
		D_scaled = stdSlr.transform(visual_words)
		self.__classifier = svm.SVC(kernel='rbf', C=1,gamma=.002).fit(D_scaled, train_labels)

		# Set 'flags'
		self.__configured = True
		
	def train(self, descriptors, labels):
		self.__classifier.fit(descriptors, labels) 

	def predict(self, descriptor):
		return self.__classifier.predict(descriptor)
	
	def getClassifier(self):
		return self.__classifier

	def BagOfWords(self, k, Train_descriptors):
		visual_words=np.zeros((len(Train_descriptors),k),dtype=np.float32)
		for i in xrange(len(Train_descriptors)):
			words=codebook.predict(Train_descriptors[i])
			visual_words[i,:]=np.bincount(words,minlength=k)
		return words, visual_words
	