import numpy as np
import cPickle
from M3S1_Evaluation import M3S1_Evaluation
from M3S1_ImageFeatureExtractor import ImageFeatureExtractor
from M3S1_Classifier import Classifier

# Pipeline class to model the different stages of our solutions
class M3S1_Pipeline:
	__featureExtractor = ImageFeatureExtractor("")
	__classifier = Classifier()
	__evaluation = None

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

	# extract features of a list of images
	def __extractAllFeatures(self, filenames, labels):
		Train_descriptors = []
		Train_label_per_descriptor = []
		for i in range(len(filenames)):
			filename=filenames[i]
			print '===' + filename + '==='
			if Train_label_per_descriptor.count(labels[i])<30:
				Train_label_per_descriptor.append(labels[i])

				print 'Reading image '+filename
				keypoints, descriptors = self.__featureExtractor.extractFeatures(filename)
				Train_descriptors.append(descriptors)

		D=Train_descriptors[0]
		L=np.array([Train_label_per_descriptor[0]]*Train_descriptors[0].shape[0])

		for i in range(1,len(Train_descriptors)):
			D=np.vstack((D,Train_descriptors[i]))
			L=np.hstack((L,np.array([Train_label_per_descriptor[i]]*Train_descriptors[i].shape[0])))        

		return D, L

	# Classifier training
	def __trainClassifier(self, descriptors, labels):
		self.__classifier.train(descriptors, labels)

	# Given an image, it predicts its label using our trained classifier
	def __predictClass(self, filename):  
		kpt,des = self.__featureExtractor.extractFeatures(filename)
		predictions = self.__classifier.predict(des)

		# Now we need to aggregate them all into a single image classification
		values, counts = np.unique(predictions, return_counts=True)
		predictedclass = values[np.argmax(counts)]

		return predictedclass

	# Classify a list of images
	def __classifyImages(self, filenames):
		predictedclassList=[];
		for i in range(len(filenames)):
			filename = filenames[i]
			predictedclass = self.__predictClass(filename)
			predictedclassList.append(predictedclass)

			print 'image '+filename+' predicted '+predictedclass

		return predictedclassList

	def run(self):
		# read train and test files
		train_image_filenames, train_labels = self.__readInput('train')
		test_images_filenames, test_labels = self.__readInput('test')

		# extract features from images and train classifier
		D, L = self.__extractAllFeatures(train_image_filenames, train_labels)
		self.__trainClassifier(D, L)

		# predict test images with classifier
		predictedclassList = self.__classifyImages(test_images_filenames)

		# assess performance
		self.__evaluation = M3S1_Evaluation(predictedclassList);

	def getEvaluation(self):
		return self.__evaluation

	def getFeatureExtractor(self):
		return self.__featureExtractor

	def getClassifier(self):
		return self.__classifier
