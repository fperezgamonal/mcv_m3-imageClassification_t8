#!/usr/bin/env python2
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

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

	def configureSVM(self, C, gamma, kernel):

		# Create and configure SVM
		if kernel == 0:
			kernel_str = 'linear'
		elif kernel == 1:
			kernel_str = 'rbf'
		elif kernel == 2:
			kernel_str = 'sigmoid'
		else:
			kernel_str = 'rbf'

		self.__classifier = svm.SVC(C=C, kernel=kernel_str, degree=3, gamma=gamma,
									coef0=0.0, shrinking=True, probability=False,
									tol=1e-3, cache_size=200, class_weight=None,
									verbose=False, max_iter=-1, decision_function_shape='ovr',
									random_state=None)

		# Set 'flags'
		self.__configured = True
		
	def train(self, descriptors, labels):
		self.__classifier.fit(descriptors, labels) 

	def predict(self, descriptor):
		return self.__classifier.predict(descriptor)
	
	def getClassifier(self):
		return self.__classifier
