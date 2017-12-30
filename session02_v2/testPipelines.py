#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 23:02:04 2017

@author: cesc
"""

import cPickle
import time
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC
from sklearn.decomposition import PCA, NMF
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import StandardScaler

from M3S2_CodeBook import CodeBook

# read the train and test files
train_images_filenames = cPickle.load(open('train_images_filenames.dat','r'))
test_images_filenames = cPickle.load(open('test_images_filenames.dat','r'))
train_labels = cPickle.load(open('train_labels.dat','r'))
test_labels = cPickle.load(open('test_labels.dat','r'))
print 'Loaded '+str(len(train_images_filenames))+' training images filenames with classes ',set(train_labels)
print 'Loaded '+str(len(test_images_filenames))+' testing images filenames with classes ',set(test_labels)


pipe = Pipeline([
		('descriptor', CodeBook()),
		('scaler', StandardScaler()),
		('classify', SVC())
])

'''
grid = GridSearchCV(pipe, cv=5, n_jobs=1, param_grid=params)
grid.fit(train_images_filenames, train_labels)

print(grid.best_params_)
'''
pipe.set_params(	descriptor__k=512,
					descriptor__numFeatures=300,
					classify__kernel="rbf",
					classify__gamma=.002,
					classify__C=1)
pipe.fit(train_images_filenames, train_labels)

accuracy = 100 * pipe.score(test_images_filenames, test_labels)

print accuracy



