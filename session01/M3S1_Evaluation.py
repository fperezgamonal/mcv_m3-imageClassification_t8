import numpy as np
import cPickle
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

# per a Conf Matrix
from sklearn.metrics import confusion_matrix
	
class M3S1_Evaluation:

	__slots__=['__TP','__TN','__FP','__FN', '__test_labels', '__predictedclass']

	#initialize vars
	def __init__(self, predictedclass):
		
		self.__TP = None
		self.__FP = None
		self.__TN = None
		self.__FN = None
		self.__test_labels = cPickle.load(open('test_labels.dat','r'))
		self.__predictedclass = predictedclass		
		
	# Accuracy
	def accuracy(self):
		return accuracy_score(self.__test_labels, self.__predictedclass)

	# Precision
	def precision(self):
		return precision_score(self.__test_labels, self.__predictedclass, average='weighted')
		
	# Recall
	def recall(self):
		return recall_score(self.__test_labels, self.__predictedclass, average='weighted')
		
	# F1 Score
	def f1Score(self):
		return f1_score(self.__test_labels, self.__predictedclass, average='weighted')
		
	# Confusion Matrix
	def confMatrix(self):
		cm = confusion_matrix(self.__test_labels, self.__predictedclass)
		
		return cm

	def printConfMatrix(self):
		cm = self.confMatrix()
		
		norm_conf = []
		for i in cm:
		    a = 0
		    tmp_arr = []
		    a = sum(i, 0)
		    for j in i:
		        tmp_arr.append(float(j)/float(a))
		    norm_conf.append(tmp_arr)
		
		fig = plt.figure()
		plt.clf()
		ax = fig.add_subplot(111)
		ax.set_aspect(1)
		ax.imshow(np.array(norm_conf), cmap=plt.cm.jet, 
		                interpolation='nearest')
		
		width, height = cm.shape
		
		for x in xrange(width):
		    for y in xrange(height):
		        ax.annotate(str(cm[x][y]), xy=(y, x), 
		                    horizontalalignment='center',
		                    verticalalignment='center')
		
		#cb = fig.colorbar(res)
		classes = sorted(set(self.__test_labels))
		plt.xticks(range(width), classes, rotation='vertical')
		plt.yticks(range(height), classes)
		plt.show()	

	def tp (self):
		cm = self.confMatrix()
		TP = 0;
		for i in range(np.size(cm[0])):
			TP = TP + cm[i, i]
		self.__TP = TP
		return  self.__TP
	
	# False Positives
	def fp (self):
		self.__FP = (self.__TP/self.precision()) - self.__TP
		return self.__FP

	# True Negatives
	def tn (self):
		self.__TN = (self.__TP/self.recall()) - self.__TP
		return self.__TN

	# False Negatives
	def fn (self):
		self.__FN = ((self.__TP + self.__TN)/self.accuracy()) - (self.__TP + self.__TN + self.__FP)
		return self.__FN
		
	# print all evaluation data
	def printEvaluation(self):
		print "confusion matrix = " 
		print self.confMatrix()
		
		print "accuracy = " +  str(self.accuracy())
		print "precision = " + str(self.precision())
		print "recall = " + str(self.recall())
		print "f1 Score = " + str(self.f1Score())
		
		print "TP = " + str(self.tp())
		print "TN = " + str(self.tn())
		print "FP = " + str(self.fp())
		print "FN = " + str(self.fn())
		
		self.printConfMatrix()
		

