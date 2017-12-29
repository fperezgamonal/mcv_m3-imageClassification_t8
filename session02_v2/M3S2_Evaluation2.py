import numpy as np
import cPickle
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import cluster

# per a Conf Matrix
from sklearn.metrics import confusion_matrix
	
class Evaluation:

	__slots__=['__TP','__TN','__FP','__FN', '__test_labels', 'predictedclass', '__visual_words_test']

	#initialize vars
	def __init__(self, predictedclass, visual_words_test):
		
		self.__Hits = None
		self.__Misses = None
		self.__test_labels = cPickle.load(open('test_labels.dat','r'))
		self.__visual_words_test = visual_words_test	
		self.__predictedclass = predictedclass
		
		self.__TotalGoldLabel = 0
		self.__TotalPredicted = 0
		self.__TP = None
		self.__cm = None
	# Accuracy
	def accuracy(self):
		#print 'Testing the SVM classifier...'
		return 100*clf.score(stdSlr.transform(self.__visual_words_test), self.__test_labels)

	# Precision
	def precision(self):
		for fil in range(np.size(self.__cm[0])):
			for i in range(np.size(self.__cm[0])):
				self.__TotalPredicted  = self.__TotalPredicted + self.__cm[fil, i]
			precision = precision + (self.__TP/self.__TotalPredicted) 
			
		return precision/ np.size(self.__cm[0])
		

		# Recall
	def recall(self):
		for col in range(np.size(self.__cm[0])):
			for i in range(np.size(self.__cm[0])):
				self.__TotalGoldLabel  = self.__TotalGoldLabel + self.__cm[i, col]
			recall = recall + (self.__TP/self.__TotalPredicted) 
		
		return recall/ np.size(self.__cm[0])		
		
	
	# F1 Score
	def f1Score(self):
	#	return f1_score(self.__test_labels, self.__predictedclass, average='weighted')
		return 2*(self.precision() * self.recall()) / (self.precision() + self.recall())
		
	# Confusion Matrix
	def confMatrix(self):
		self.__cm =  confusion_matrix(self.__test_labels, self.__predictedclass)
		return self.__cm
		
	def printConfMatrix(self):
		cm = self.__cm
		
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

	def HitsAndMisses (self):
		hits = 0;
		for i in range(np.size(self.__cm[0])):
			hits = hits + self.__cm[i, i]
		self.__Hits = hits
		
		suma = np.sum(self.__cm)
		self.__Misses = suma - self.__Hits
		
		return  self.__Hits, self.__Misses
	
	# print all evaluation data
	def printEvaluation(self):
		init=time.time()
		
		print "confusion matrix = " 
		print self.confMatrix()
		
		hm = self.HitsAndMisses()
		print "Hits = " + str(hm[0])
		print "Misses = " + str(hm[1])
	
		print "accuracy = " +  str(self.accuracy())
	#	print "precision = " + str(self.precision())
	#	print "recall = " + str(self.recall())
	#	print "f1 Score = " + str(self.f1Score())
		
			
		self.printConfMatrix()
		
		end=time.time()
		print 'Evaluation done in '+str(end-init)+' secs.'
		

