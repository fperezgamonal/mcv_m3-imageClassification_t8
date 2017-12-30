#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import os
import cPickle
import time
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import  SVC
from sklearn.preprocessing import StandardScaler

from M3S2_CodeBook import CodeBook

def saveXVal(grid):
	root_folder = os.path.join(os.pardir, "PreComputed_Params/XVal/")
	if not os.path.isdir(root_folder):
		os.mkdir(root_folder)
	
	filename = os.path.join(root_folder, "xval_", str(time.time()) , ".pkl")
	cPickle.dump(grid, open(filename, "wb"))

# read the train and test files
train_images_filenames = cPickle.load(open('train_images_filenames.dat','r'))
test_images_filenames = cPickle.load(open('test_images_filenames.dat','r'))
train_labels = cPickle.load(open('train_labels.dat','r'))
test_labels = cPickle.load(open('test_labels.dat','r'))
print 'Loaded '+str(len(train_images_filenames))+' training images filenames with classes ',set(train_labels)
print 'Loaded '+str(len(test_images_filenames))+' testing images filenames with classes ',set(test_labels)

pipe = Pipeline(steps=[
		('descriptor', CodeBook()),
		('scaler', StandardScaler()),
		('classify', SVC())])

params = dict(	descriptor__k=range(10,30,10),
					descriptor__numFeatures=[300],
					classify__kernel=["rbf"],
					classify__gamma= [.002],
					classify__C=[1])



start = time.time()
grid = GridSearchCV(pipe, cv=2, n_jobs=1, param_grid=params)
grid.fit(train_images_filenames, train_labels)
end = time.time()

saveXVal()

print(grid.best_params_)

print("All done in ", str(end-start), " seconds.")


print("Best parameters set found on development set:")
print()
print(grid.best_params_)
print()
print("Grid scores on development set:")
print()
means = grid.cv_results_['mean_test_score']
stds = grid.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, grid.cv_results_['params']):
	print("%0.3f (+/-%0.03f) for %r"% (mean, std * 2, params))
print()