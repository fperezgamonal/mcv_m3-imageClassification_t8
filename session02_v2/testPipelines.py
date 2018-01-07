#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import cPickle
import time
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from M3S2_CodeBook import CodeBook
from M3S2_Evaluation import Evaluation

# read the train and test files
train_images_filenames = cPickle.load(open('train_images_filenames.dat','r'))
test_images_filenames = cPickle.load(open('test_images_filenames.dat','r'))
train_labels = cPickle.load(open('train_labels.dat','r'))
test_labels = cPickle.load(open('test_labels.dat','r'))
print 'Loaded '+str(len(train_images_filenames))+' training images filenames with classes ',set(train_labels)
print 'Loaded '+str(len(test_images_filenames))+' testing images filenames with classes ',set(test_labels)

start = time.time()

pipe = Pipeline([
		('descriptor', CodeBook()),
		('scaler', StandardScaler()),
		('classify', SVC(verbose=True))
])

pipe.set_params(	descriptor__k=512,
					descriptor__descType="SpatialPyramids",
					descriptor__numFeatures=1000,
					classify__kernel=CodeBook.pyramidMatchKernel)

pipe.fit(train_images_filenames, train_labels)



predicted_labels = pipe.predict(test_images_filenames)

end = time.time()
print("All done in ", str(end-start), " seconds.")

evaluation = Evaluation(predicted_labels)
evaluation.printEvaluation()

#accuracy = 100 * pipe.score(test_images_filenames, test_labels)
#print accuracy



