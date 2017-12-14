import numpy as np
import cPickle
from operator import truediv


# per a Conf Matrix
from sklearn.metrics import confusion_matrix

# per a KFold Cross Validation
from sklearn.model_selection import KFold
	
class M3S1_Evaluation:

	__slots__=['__TP','__TN','__FP','__FN', '__test_labels', '__predictedclass']

	#initialize vars
	def __init__(self, TP, TN, FP, FN, test_labels, predictedclass):
		
		self.__TP = None
		self.__FP = None
		self.__TN = None
		self.__FN = None
		self.__test_labels = cPickle.load(open('test_labels.dat','r'))
		self.__predictedclass = predictedclass		
		
	# Accuracy
	def accuracy(self):
		a = self.__TP+self.__TN
		b = self.__TP+self.__TN+self.__FP+self.__FN
		return truediv(a,b)

	# Precision
	def precision(self):
		a = self.__TP
		b = self.__TP+self.__FP
		return truediv(a,b)
		
	# Recall
	def recall(self):
		a = self.__TP
		b = self.__TP+self.__TN
		return truediv(a,b)
		
	# F1 Score
	def f1Score(self):
		a = 2 * self.precision() * self.recall()
		b = self.precision() + self.recall()
		return truediv(a,b)
		
	# Confusion Matrix
	def confMatrix(self):
		cm = confusion_matrix(self.__test_labels, self.__predictedclass)	
		
		TP = 0;
		TN = 0;
		FN = 0;
		FP = 0;
		sz= np.size(cm[0])
		print  'size = ' + str(sz)
		for i in range(sz):
			# True Positives
			TP = cm[i, i]
			
			# False Negatives
			fn_mask = np.zeros(cm.shape)
			fn_mask[i, :] = 1
			fn_mask[i, i] = 0
			FN = np.sum(np.multiply(cm, fn_mask))
			
			# False Positives
			fp_mask = np.zeros(cm.shape)
			fp_mask[:, i] = 1
			fp_mask[i, i] = 0
			FP = np.sum(np.multiply(cm, fp_mask))

			# True Negatives
			tn_mask = 1 - (fn_mask + fp_mask)
			tn_mask[i, i] = 0
			TN1 = np.sum(np.multiply(cm, tn_mask))
		
		self.__TP = TP
		self.__FP = FP
		self.__TN = TN
		self.__FN = FN
		
		return cm
		

	# KFold crossValidation	
	def crossValidation (self):
	
		# data is an array with our already pre-processed dataset examples
		kf = KFold(n_splits=8)
		sum = 0
		
		data = zip(self.__test_labels,self.__predictedclass)
		#from nltk.classify import NaiveBayesClassifier

		for train, test in kf.split(data):
			train_data = np.array(data)[train]
			
			aux = map(list, (zip(*train_data)))
			
			
			test_data = np.array(data)[test]
			
			#classifier = NaiveBayesClassifier.train(train_data)
		
			# aux[0] => labels
			# aux[1] =>  predictedclass
			
			#parameters of train classifier KNN: test_images_filenames[i], detector, classifier
			classifier = self.trainClassifier(aux[0], aux[1])
			
			#sum += nltk.classify.accuracy(classifier, test_data)
			
			# recalculate TP TN FP FN (the sane as  in cross validation)
			TP = 0;
			TN = 0;
			FN = 0;
			FP = 0;
			sz= np.size(cm[0])
			print  'size = ' + str(sz)
			for i in range(sz):
				# True Positives
				TP = cm[i, i]
				
				# False Negatives
				fn_mask = np.zeros(cm.shape)
				fn_mask[i, :] = 1
				fn_mask[i, i] = 0
				FN = np.sum(np.multiply(cm, fn_mask))
				
				# False Positives
				fp_mask = np.zeros(cm.shape)
				fp_mask[:, i] = 1
				fp_mask[i, i] = 0
				FP = np.sum(np.multiply(cm, fp_mask))
				
				# True Negatives
				tn_mask = 1 - (fn_mask + fp_mask)
				tn_mask[i, i] = 0
				TN1 = np.sum(np.multiply(cm, tn_mask))
			
			a = TP+TN
			b = TP+TN+FP+FN
			sum +=  a/b
			
		average = truediv(sum,8)
			
		return average 
		
	# print all evaluation data
	def printEvaluation(self):
		print "confusion matrix = " 
		print self.confMatrix()
		
		print "accuracy = " +  str(self.accuracy())
		print "precision = " + str(self.precision())
		print "recall = " + str(self.recall())
		print "f1 Score = " + str(self.f1Score())
		print "cross validation " + str(self.crossValidation())
		
		print "TP = " + str(self.__TP)
		print "TN = " + str(self.__TN)
		print "FP = " + str(self.__FP)
		print "FN = " + str(self.__FN)
	
	def trainClassifier(D, L):
		# Train a k-nn classifier
		
		print 'Training the knn classifier...'
		myknn = KNeighborsClassifier(n_neighbors=5,n_jobs=-1)
		myknn.fit(D,L)
		print 'Done!'

		return myknn