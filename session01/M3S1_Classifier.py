#!/usr/bin/env python2
from sklearn.neighbors import KNeighborsClassifier

# Classifier class that encapsulates a generic classifier
# An instance of this class is expected to be configured first in order
# to impersonate a given classifier (such as KNN). 
# After configuration, it can be train()'ed with a list of descriptors and a 
# list of their labels (np.arrays). 
# We can also predict() a descriptor using the given classifier
class Classifier:
    configured = False
    type = None
    knn = None
    
    def getType(self):
        return self.type
    
    def isConfigured(self):
        return self.configured

    # each different type of classifier implemented in this class
    # must have its own configuration method in order to instantiate
    # the classifier with a set of parameters
    # configuration of the classifier is mandatory (see self.configured usage)
    def configureKNN(self, numNeighbors=5, numJobs=-1):
        assert(not self.configured)
        
        self.knn = KNeighborsClassifier(n_neighbors=numNeighbors, n_jobs=numJobs)  
        self.configured = True
        self.type = 'KNN'
        
    def train(self, descriptors, labels):
        assert(self.configured)
        
        if self.type == 'KNN':
            print 'Training the knn classifier...'
            self.knn.fit(descriptors, labels) 
        # training of other types of classifiers goes here
        # else if self.type == 'Whatever'
        #   ...
            
        print 'Done!'
        
    def predict(self, descriptor):
        assert(self.configured)
        
        if self.type == 'KNN':
            return self.knn.predict(descriptor)
        # prediction for other classifiers goes here ..
        # else if self.type == 'Whatever':
        #   ...
        
        return None