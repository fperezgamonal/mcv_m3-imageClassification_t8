import cv2
import numpy as np
import cPickle
import time
from sklearn.neighbors import KNeighborsClassifier

class M3S1_Main:
	start = time.time()

	# read train and test files
	def readInput(type):   
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
		return filenames, labels

	# Extract SIFT features from an image
	def extractFeatures(filenames, labels):
		# create the SIFT detector object
		(major, minor, _) = cv2.__version__.split(".")
		
		# create SIFT detector object
		if major >= 3:
			SIFTdetector = cv2.xfeatures2d.SIFT_create()
		else:
			SIFTdetector = cv2.SIFT(nfeatures=100)
		
		# read the just 30 train images per class
		# extract SIFT keypoints and descriptors
		# store descriptors in a python list of numpy arrays
		
		Train_descriptors = []
		Train_label_per_descriptor = []
		
		for i in range(len(filenames)):
			filename=filenames[i]
			if Train_label_per_descriptor.count(labels[i])<30:
				print 'Reading image '+filename
				ima=cv2.imread(filename)
				gray=cv2.cvtColor(ima,cv2.COLOR_BGR2GRAY)
				kpt,des=SIFTdetector.detectAndCompute(gray,None)  
				Train_descriptors.append(des)
				Train_label_per_descriptor.append(labels[i])
				print str(len(kpt))+' extracted keypoints and descriptors'
		
		# Transform everything to numpy arrays
	   
		D=Train_descriptors[0]
		L=np.array([Train_label_per_descriptor[0]]*Train_descriptors[0].shape[0])
		
		for i in range(1,len(Train_descriptors)):
			D=np.vstack((D,Train_descriptors[i]))
			L=np.hstack((L,np.array([Train_label_per_descriptor[i]]*Train_descriptors[i].shape[0])))

		return SIFTdetector, D, L

	# k-NN classifier
	def trainClassifier(D, L):
		# Train a k-nn classifier
		
		print 'Training the knn classifier...'
		myknn = KNeighborsClassifier(n_neighbors=5,n_jobs=-1)
		myknn.fit(D,L)
		print 'Done!'

		return myknn

	#Testing k-NN classifier
	def predictClass(filename, detector, classifier):  
		ima=cv2.imread(filename)
		gray=cv2.cvtColor(ima,cv2.COLOR_BGR2GRAY)
		kpt,des=detector.detectAndCompute(gray,None)
		predictions = classifier.predict(des)
		
		# Now we need to aggregate them all into a single image classification
		values, counts = np.unique(predictions, return_counts=True)
		predictedclass = values[np.argmax(counts)]
		
		return predictedclass


	def INIT()
		# read train and test files
		train_image_filenames, train_labels = readInput('train')
		test_images_filenames, test_labels = readInput('test')

		# Extract SIFT features from an image
		detector, D, L = extractFeatures(train_image_filenames, train_labels)

		# k-NN classifier
		classifier = trainClassifier(D, L)

		# =======================
		# Test with performance evaluation
		# Get all test data and predict  their labels

		numtestimages=0
		numcorrect=0

		for i in range(len(test_images_filenames)):
			#filename=test_images_filenames[i]
			#Testing k-NN classifier
			predictedclass[i] = predictClass(test_images_filenames[i], detector, classifier)

			print 'image '+filename+' was from class '+test_labels[i]+' and was predicted '+predictedclass[i]
			numtestimages+=1
			if predictedclass[i]==test_labels[i]:
				numcorrect+=1

		print 'Final accuracy: ' + str(numcorrect*100.0/numtestimages)

		end=time.time()
		
		ret = print 'Done in '+str(end-start)+' secs.'
		print ret
		
		return ret

	## 30.48% in 302 secs.