import numpy as np
import time
import warnings
import os.path
import cPickle
from M3S2_Evaluation import Evaluation
from M3S2_ImageFeatureExtractor import ImageFeatureExtractor
from M3S2_Classifier import Classifier
from M3S2_CodeBook import  CodeBook
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn import cluster

# Classifiers imports (we need to initialise them for GridSearchCV)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Pipeline class to model the different stages of our solutions
class Pipeline:
	__slots__=['__featureExtractor', '__classifier', '__evaluation', '__codebook']

	def __init__(self, recordData=False, useRecordedData=False, recordedDataName=''):
		self.__featureExtractor = ImageFeatureExtractor("", recordData, useRecordedData, recordedDataName)
		self.__classifier = Classifier()
		self.__evaluation = None
		self.__codebook = CodeBook()
		
	# read either train or test files
	def __readInput(self, type):   
		try:
			if type == 'train':
				#train_image_filenames
				filenames = cPickle.load(open('train_images_filenames.dat','r')) 
				#train_labels
				labels = cPickle.load(open('train_labels.dat','r'))
				print 'Loaded '+str(len(filenames))+' training images filenames with classes ',set(labels) 
			elif type == 'test':
				#test_images_filenames
				filenames = cPickle.load(open('test_images_filenames.dat','r')) 
				#test_labels
				labels = cPickle.load(open('test_labels.dat','r'))
				print 'Loaded '+str(len(filenames))+' testing images filenames with classes ',set(labels)
		except ValueError:
			filenames = []
			labels = []
			print 'Nothing was loaded'
		return filenames, labels

	# Save features to disk so they do not have to be recomputed
	def __saveFeaturesToDisk(self, descriptors,  filename):
		with open(filename, 'w') as f:
			cPickle.dump([descriptors], f)


	# Load features from disk
	def __loadFeaturesFromDisk(self, filename):
		with open(filename) as f:
			D, L = cPickle.load(f)
			return D, L			# tuple

	# Define similar functions for storing the models
	# Note: ideally, in a long-time project, 2 functions would be enough (not 4)

	def __saveModelToDisk(self, model, filename):
		with open(filename, 'w') as f:
			cPickle.dump(model, f)

	def __loadModelFromDisk(self, filename):
		with open(filename) as f:
			model = cPickle.load(f)
			return model

	# extract features of a list of images
	def __extractAllFeatures(self, filenames, labels):
		Train_descriptors = []
		Train_label_per_descriptor = []
		for i in range(len(filenames)):
			filename=filenames[i]
			#print '===' + filename + '==='
			#if Train_label_per_descriptor.count(labels[i])<30:
			print 'Reading image '+filename
			keypoints, descriptors = self.__featureExtractor.extractFeatures(filename)
			Train_descriptors.append(descriptors)
			Train_label_per_descriptor.append(labels[i])

		size_descriptors=Train_descriptors[0].shape[1]
		D=np.zeros((np.sum([len(p) for p in Train_descriptors]),size_descriptors),dtype=np.uint8)
		startingpoint=0
		#L=np.array([Train_label_per_descriptor[0]]*Train_descriptors[0].shape[0])

		for i in range(len(Train_descriptors)):
			D[startingpoint:startingpoint+len(Train_descriptors[i])]=Train_descriptors[i]
			startingpoint+=len(Train_descriptors[i])
	
		return D, Train_descriptors

	# Classifier training
	def __trainClassifier(self, descriptors, labels):
		self.__classifier.train(descriptors, labels)
		return self.__classifier

	# Given an image, it predicts its label using our trained classifier
	def __predictClass(self, filename):#, img_stored_feature):
		#if not img_stored_feature:  # if empty (features were not loaded)
		kpt, des = self.__featureExtractor.extractFeatures(filename)
		predictions = self.__classifier.predict(des)
	#	else:  # use computed feature to predict
	#		predictions = self.__classifier.predict(img_stored_feature)

		# Now we need to aggregate them all into a single image classification
		values, counts = np.unique(predictions, return_counts=True)
		predictedclass = values[np.argmax(counts)]

		return predictedclass

	# Classify a list of images
	def __classifyImages(self, filenames):#, feat_filename):
		predictedclassList=[]
		#img_feature = []
		for i in range(len(filenames)):
			filename = filenames[i]
			#if os.path.isfile(feat_filename):
			#	imf_feature = 0

			predictedclass = self.__predictClass(filename)#, img_feature)
			predictedclassList.append(predictedclass)

			print 'image '+filename+' predicted '+predictedclass
		
		return predictedclassList

	def run(self, feat_type, feat_num, clf_type, clf_params):
		# Define filenames to look for already-computed stuff
		# 1) features (for 'train' and 'test')
		feat_filename = "PreComputed_Params/Features/features_train_" +\
						feat_type + '_' + str(feat_num) + ".pkl"
		feat_filename = os.path.join(os.pardir, feat_filename)

		# 2) model (obviously for 'train' only, inference is not stored!)
		if clf_type == 'KNN':
			clf_str_params = "_NN" + str(clf_params['n_neighbors'])
		elif clf_type == 'RForest':
			clf_str_params = "_NT" + str(clf_params['n_estimators'])
		elif clf_type == 'SVM':
			clf_str_params = "_C" + str(clf_params['C']) + "_gm" +\
				str(clf_params['gamma']) + "_kernel_" +\
							 clf_params['kernel']
		else: # Bayes
			clf_str_params = ""

		model_filename = "PreComputed_Params/Models/model_" + feat_type +\
			'_' + str(feat_num) + '-' + clf_type + clf_str_params + ".pkl"
		model_filename = os.path.join(os.pardir, model_filename)

		if os.path.isfile(feat_filename):
			# Load features
			print "Same features located in disk. Loading them..."
			D, L = self.__loadFeaturesFromDisk(feat_filename)
			print "Features loaded successfully"

		else:
			print "These features have not been computed before."
			# read train and test files*
			# only if we have not computed the same features before
			print "Loading training data and computing features..."
			train_image_filenames, train_labels = self.__readInput('train')
			# extract features from images and train classifier*
			# * only if they have not been already computed
			D, Train_descriptors = self.__extractAllFeatures(train_image_filenames, train_labels)

			# Save features for later usage
			self.__saveFeaturesToDisk(D, feat_filename)

		#compute the codebook
		k = 512
		codebook = self.__codebook.CB(k)
		 
		# get train visual word encoding
		print 'Getting Train BoVW representation'#
		self.__classifier.BagOfWords(k, Train_descriptors)
		
		# train ...
		
		if os.path.isfile(model_filename):
			# Load model
			warnings.warn("WARNING: this model was already trained. Loading it from disk...")
			self.__classifier = self.__loadModelFromDisk(model_filename)
			print "Model loaded successfully"

		else:
			print "This model was not trained before. Starting training..."
			# Train classifier (model)*
			# Yet again: only if a model with the same params has not been created yet
			start = time.time()
			self.__classifier = self.__trainClassifier(D, L)
			finish = time.time()
			elapsed = finish - start
			print 'Done in '+str(elapsed)+' secs.'
			# Models can be very large in memory, consider if it is useful to store it
			# Check if the time training is larger than, say: 10 minutes?
			# NOTE: this is quite contradictory as for larger training times, we'd like
			# to load the model, saving > 10 minutes but then we will be storing models of Gb's!!

			maxTrainSecs = 600  # 10 minutes
			if elapsed < maxTrainSecs:
				print "Saving model to disk as it took less than {%d} to train the classifier (rel. small model)".format()
				# Store model to disk
				self.__saveModelToDisk(self.__classifier, model_filename)

		# predict test images with classifier
		# (inside we can check if the features have been computed)
		# feat_filename_test = "PreComputed_Params/Features/features_test_" + \
		# 						 feat_type + '_' + str(feat_num) + ".pkl"
		# feat_filename_test = os.path.join(os.pardir, feat_filename_test)

		
		print 'Getting Test BoVW representation'
		init=time.time()
		test_images_filenames, test_labels = self.__readInput('test')
		visual_words_test=np.zeros((len(test_images_filenames),k),dtype=np.float32)
		for i in range(len(test_images_filenames)):
			kpt, des = self.__featureExtractor.extractFeatures(test_images_filenames)
			words=codebook.predict(des)
			visual_words_test[i,:]=np.bincount(words,minlength=k)
		
		end=time.time()
		print 'Done in '+str(end-init)+' secs.'
		
		#predictedclassList = self.__classifyImages(test_images_filenames)#, feat_filename_test)
		
		# assess performance
		self.__evaluation = Evaluation(visual_words_test)

	def KFoldCrossValidate(self, clf_params, clf_type, feat_type, numFeats):
		# clf_params is a dictionary containing the parameters and values
		# of the classifier to test in CV

		# Check if the features have been computed and avoid reloading,etc.
		feat_filename = 'PreComputed_Params/Features/features_train_' + feat_type + '_' + str(numFeats) + '.pkl'
		# Store it in the repo root folder so it does not depend on being on
		# session01 or session0X
		feat_filename = os.path.join(os.pardir, feat_filename)

		if os.path.isfile(feat_filename):
			# Load features
			D,L = self.__loadFeaturesFromDisk(feat_filename)

		else:
			# read train and test files, extract features and save them
			train_image_filenames, train_labels = self.__readInput('train')
			D, Train_descriptors= self.__extractAllFeatures(train_image_filenames, train_labels)

			# Save features for later usage
			self.__saveFeaturesToDisk(D, feat_filename)

		kfolds = StratifiedKFold(n_splits=5, shuffle=False, random_state=42)
		# Initialise the correct type of classifier:
		if clf_type == 'KNN':
			grid = GridSearchCV(KNeighborsClassifier(), param_grid=clf_params, cv=kfolds,
								scoring='accuracy', error_score=0,
								n_jobs=-1, verbose=10)
		elif clf_type == 'GaussNBayes':
			grid = GridSearchCV(GaussianNB(), param_grid=clf_params, cv=kfolds,
								scoring='accuracy', error_score=0,
								n_jobs=-1, verbose=10)
		elif clf_type == 'MultinomialNBayes':
			grid = GridSearchCV(MultinomialNB(), param_grid=clf_params, cv=kfolds,
								scoring='accuracy', error_score=0,
								n_jobs=-1, verbose=10)
		elif clf_type == 'RForest':
			grid = GridSearchCV(RandomForestClassifier(), param_grid=clf_params, cv=kfolds,
								scoring='accuracy', error_score=0,
								n_jobs=-1, verbose=10)
		elif clf_type == 'SVM':
			grid = GridSearchCV(SVC(), param_grid=clf_params, cv=kfolds,
								scoring='accuracy', error_score=0,
								n_jobs=-1, verbose=10)
			# IMPORTANT!: normalize features
			stdSlr = StandardScaler().fit(D)
			D = stdSlr.transform(D)
		else:
			# Load KNN as default (to avoid errors)
			grid = GridSearchCV(KNeighborsClassifier(), param_grid=clf_params, cv=kfolds,
								scoring='accuracy', error_score=0,
								n_jobs=-1, verbose=10)

		
		#cv = KFold(len(L), k, shuffle=True)
		#accs = cross_val_score(self.getClassifier().getClassifier(), D, L, cv=cv, scoring='accuracy')

		# Return grid (only needs to be train as grid.fit(X, y)
		# We can then, delete the loop of 'cross_val.py' and just create the needed dictionary
		grid.fit(D, L)

		print("Best parameters: %s Accuracy: %0.2f" % (grid.best_params_, grid.best_score_))
		# In case we need the extra params: best.estimator or best.index
		return grid


	def getEvaluation(self):
		return self.__evaluation

	def getFeatureExtractor(self):
		return self.__featureExtractor

	def getClassifier(self):
		return self.__classifier
