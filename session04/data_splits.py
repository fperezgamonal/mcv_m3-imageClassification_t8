#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from keras.preprocessing.image import ImageDataGenerator, array_to_img
import os
import numpy as np

DATASET_DIR = '../../Databases/MIT_split'

def createSplit(num_images):
	BATCH_SIZE = num_images
	SPLIT_DIR = DATASET_DIR + "/train_split_" + str(BATCH_SIZE) + "/"
	 
	train_datagen = ImageDataGenerator()
	
	classes = ['coast','forest','highway','inside_city','mountain','Opencountry','street','tallbuilding']
	if not os.path.exists(SPLIT_DIR):
		os.mkdir(SPLIT_DIR)
	for c in classes:
		if not os.path.exists(SPLIT_DIR + c):
			os.mkdir(SPLIT_DIR + c)
	
	train_generator = train_datagen.flow_from_directory(
	        DATASET_DIR+'/train',  # this is the target directory
	        batch_size=BATCH_SIZE,
	        classes = classes,
	        shuffle=True,
	        class_mode='sparse',
	        seed=43)
	
	ci = train_generator.class_indices
	
	X,Y = train_generator.next()
	for i in range(len(X)):
		category = ci.keys()[ci.values().index(Y[i])]
		img = array_to_img(X[i])
		img.save(os.path.join(SPLIT_DIR,category,str(i) + ".jpg"))


for n in range(400,1800,400):
	createSplit(n)


