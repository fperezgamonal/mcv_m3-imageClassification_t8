import numpy as np
import cPickle
import time

# per a Conf Matrix
from session1 import test_labels, test_images_filenames
from sklearn.metrics import confusion_matrix

# per a KFold Cross Validation

from M3S1_Main import trainClassifier

from sklearn.model_selection import KFold
	
class M3S1_Evaluation:

	__slots__=['__TP','__TN','__FP','__FN']

	#initialize vars
	def __init__(self, TP, TN, FN, FP, test_labels, predictedclass):
		
		self.__TP = None
		self.__FP = None
		self.__TN = None
		self.__FN = None
		confMatrix(test_labels, predictedclass)	
		
	# Accuracy
	def accuracy(self)
		a = self.__TP+self.__TN
		b = self.__TP+self.__TN+self.__FP+self.__FN
		return a/b

	# Precision
	def precision(self)
		a = self.__TP
		b = self.__TP+self.__FP
		return a/b
		
	# Recall
	def recall(self)
		a = self.__TP
		b = self.__TP+self.__TN
		return a/b
		
	# F1 Score
	def f1Score(self)
		a = 2 * precision(self) * recall(self)
		b = precision(self) + recall(self)
		return a/b
		
	# Confusion Matrix
	def confMatrix(test_labels, predictedclass)	

		cm = confusion_matrix(test_labels, predictedclass)	
		
		TP = 0;
		TN = 0;
		FN = 0;
		FP = 0;
		for i in range(np.size(cm[0]):
			TP += cm[i, i] 
			FP += np.sum(cm, axis=0)[i] - cm[i, i]  #The corresponding column for class_i - TP
			FN += np.sum(cm, axis=1)[i] - cm[i, i] # The corresponding row for class_i - TP
			TN += np.size(cm) - TP -FP -TN

		self.__TP = TP
		self.__FP = FP
		self.__TN = TN
		self.__FN = FN
		
		return cm
		

	# KFold crossValidation
	def crossValidation (test_labels, predictedclass)
		# data is an array with our already pre-processed dataset examples
		kf = KFold(n_splits=8)
		sum = 0
		data = zip(test_labels,predictedclass)
		from nltk.classify import NaiveBayesClassifier

		for train, test in kf.split(data):
			train_data = np.array(data)[train]
			test_data = np.array(data)[test]
			#classifier = NaiveBayesClassifier.train(train_data)
			classifier = M3S1_Main.trainClassifier(train_data)
			#sum += nltk.classify.accuracy(classifier, test_data)
			
			# recalculate TP TN FP FN (the sane as  in cross validation)
			cm= confMatrix(test_labels, predictedclass)	
			TP = 0;
			TN = 0;
			FN = 0;
			FP = 0;
			for i in range(np.size(cm[0]):
				TP += cm[i, i] 
				FP += np.sum(cm, axis=0)[i] - cm[i, i]  
				FN += np.sum(cm, axis=1)[i] - cm[i, i] 
				TN += np.size(cm) - TP -FP -TN

			a = TP+TN
			b = TP+TN+FP+FN
			sum +=  a/b
			
		average = sum/8
			
		return average 
		
	# ROC Curves
	def rocCurves()

	# print all evaluation data
	def printEvaluation(self, test_labels, predictedclass)
		print "accuracy " + accuracy(self) 
		print "precision " + precision(self)
		print "recall " + recall(self)
		print "f1 Score " + f1Score(self)
		print "confusion matrix " + confMatrix(test_labels, predictedclass)	
		print "cross validation " + crossValidation(test_labels)
		#print "roc curves " + rocCurves()
	

