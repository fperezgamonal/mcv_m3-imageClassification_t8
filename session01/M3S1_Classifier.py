#!/usr/bin/env python2
from sklearn.neighbors import KNeighborsClassifier

# Classifier class that encapsulates a generic classifier
# An instance of this class is expected to be configured first in order
# to impersonate a given classifier (such as KNN). 
# After configuration, it can be train()'ed with a list of descriptors and a 
# list of their labels (np.arrays). 
# We can also predict() a descriptor using the given classifier
class Classifier:
	
	__slots__=['__configured', '__type', '__knn']

	#initialize vars
	def __init__(self):
		self.__configured = False
		self.__type = None
		self.__knn = None


	def getType(self):
		return self.__type

	def isConfigured(self):
		return self.__configured

    # each different type of classifier implemented in this class
    # must have its own configuration method in order to instantiate
    # the classifier with a set of parameters
    # configuration of the classifier is mandatory (see self.__configured usage)
	def configureKNN(self, numNeighbors, numJobs):
		assert(not self.__configured)

		self.__knn = KNeighborsClassifier(n_neighbors=numNeighbors, n_jobs=numJobs)   
		self.__configured = True
		self.__type = 'KNN'
        
	def train(self, descriptors, labels):
		assert(self.__configured)

		if self.__type == 'KNN':
			print 'Training the knn classifier...'
			self.__knn.fit(descriptors, labels) 
		# training of other types of classifiers goes here
		# else if self.__type == 'Whatever'
		#   ...
		print 'Done!'
        
	def predict(self, descriptor):
		assert(self.__configured)

		if self.__type == 'KNN':
			return self.__knn.predict(descriptor)
		# prediction for other classifiers goes here ..
		# else if self.__type == 'Whatever':
		#   ...

		return None