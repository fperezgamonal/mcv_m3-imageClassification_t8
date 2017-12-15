import numpy as np
import cPickle
import time
import cv2
from M3S1_Evaluation import M3S1_Evaluation
from M3S1_ImageFeatureExtractor import ImageFeatureExtractor
from M3S1_Classifier import Classifier


class M3S1_Main:
	# read train and test files
	def __init__(self):
		self
		
	def readInput(self,type):   
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

	# Extract SIFT features from an image
	def extractAllFeatures(self,filenames, labels, featureExtractor):
		Train_descriptors = []
		Train_label_per_descriptor = []
		for i in range(len(filenames)):
			filename=filenames[i]
			#print '===' + filename + '==='
			if Train_label_per_descriptor.count(labels[i])<30:
				Train_label_per_descriptor.append(labels[i])
				
				print 'Reading image '+filename
				keypoints, descriptors = featureExtractor.extractFeatures(filename)
				Train_descriptors.append(descriptors)
				
		D=Train_descriptors[0]
		L=np.array([Train_label_per_descriptor[0]]*Train_descriptors[0].shape[0])
		
		for i in range(1,len(Train_descriptors)):
			D=np.vstack((D,Train_descriptors[i]))
			L=np.hstack((L,np.array([Train_label_per_descriptor[i]]*Train_descriptors[i].shape[0])))        
				
		return D, L

	#Testing k-NN classifier
	def predictClass(self, filename, featureExtractor, classifier):  
		kpt,des = featureExtractor.extractFeatures(filename)
		predictions = classifier.predict(des)
		return_counts=True
		
		# Now we need to aggregate them all into a single image classification
		values, counts = np.unique(predictions, return_counts)
		predictedclass = values[np.argmax(counts)]
		
		return predictedclass

	def trainClassifier(self, descriptors, labels):
		self.__classifier.train(descriptors, labels)
		
	def run(self, type):
		
		if (type=='SIFT'):
			featureExtractor = ImageFeatureExtractor(type)
			featureExtractor.configureSIFT(100)
		else: #HOG
			featureExtractor = ImageFeatureExtractor('HOG')
			featureExtractor.configureHOG()
		self.featuresFromImage(featureExtractor)
		
	def featuresFromImage(self, featureExtractor):
		# read train and test files
		train_image_filenames, train_labels = self.readInput('train')
		
		# Extract features from an image
		D, L = self.extractAllFeatures(train_image_filenames, train_labels, featureExtractor)
		self.runClassifier(D, L , featureExtractor)

	def runClassifier (self, D, L , featureExtractor):
		# k-NN classifier
		numNeighbors=5
		numJobs = -1
		classifier = Classifier()
		classifier.configureKNN(numNeighbors, numJobs)
		
		classifier.train(D, L)
		self.predictLabels(featureExtractor, classifier)
		
		# =======================
		# Test with performance evaluation
		# Get all test data and predict  their labels
	def predictLabels(self, featureExtractor, classifier):
		test_images_filenames, test_labels = self.readInput('test')

		print '==='
		numtestimages=0
		numcorrect=0
		predictedclassList=[];
		for i in range(len(test_images_filenames)):
			filename=test_images_filenames[i]
			#Testing k-NN classifier
			predictedclass = self.predictClass(test_images_filenames[i], featureExtractor, classifier)
			predictedclassList.append(predictedclass)
			
			print 'image '+filename+' was from class '+test_labels[i]+' and was predicted '+predictedclass
			numtestimages+=1
			if predictedclass==test_labels[i]:
				numcorrect+=1
		self.evaluate(predictedclassList)
		
	def evaluate (self, predictedclassList):
		# M3S1_Evaluation s'ha de passar "test_labels" i "predictedclassList"
		eval = M3S1_Evaluation(predictedclassList);
		eval.printEvaluation()

		

		

		#return ret

## 30.48% in 302 secs.

if __name__ == "__main__":
	
	start = time.time()
	
	aux = M3S1_Main()
	aux.run("HOG")
	aux.run("SIFT")
	
	end=time.time()
	ret =  'Done in '+str(end-start)+' secs.'
	print ret
	