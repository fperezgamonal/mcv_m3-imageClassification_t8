from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn import cluster

# Classifier class that encapsulates a generic classifier
# An instance of this class is expected to be configured first in order
# to impersonate a given classifier (such as KNN). 
# After configuration, it can be train()'ed with a list of descriptors and a 
# list of their labels (np.arrays). 
# We can also predict() a descriptor using the given classifier
class CodeBook:		#compute the codebook
	__slots__=['__k']

	#initialize vars
	def __init__(self, k):	
		self.__k = k
	
	def CB (self):
		print 'Computing kmeans with '+str(self.__k)+' centroids'
		init=time.time()
		codebook = cluster.MiniBatchKMeans(n_clusters=self.__k, verbose=False, batch_size=self.__k * 20,compute_labels=False,reassignment_ratio=10**-4,random_state=42)
		codebook.fit(D)
		cPickle.dump(codebook, open("codebook.dat", "wb"))
		end=time.time()
		print 'Done in '+str(end-init)+' secs.'
		return codebook
	