from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

# Evaluation metrics
# Accuracy
def accuracy(y_true, y_pred):
	return accuracy_score(y_true, y_pred)

# Precision
def precision(y_true, y_pred):
	return precision_score(y_true, y_pred, average='weighted')

# Recall
def recall(y_true, y_pred):
	return recall_score(y_true, y_pred, average='weighted')

# F1-score
def f1score(y_true, y_pred):
	return f1_score(y_true, y_pred, average='weighted')

# Confusion matrix
def confusionMatrix(y_true, y_pred):
	return confusion_matrix(y_true, y_pred)

def HitsAndMisses(cm):
	#TP = 0
	# for i in range(np.size(cm[0])):
	# 	TP = TP + cm[i, i]
	# self.__TP = TP
	# TP = trace (sum of) matrix diagonal elements
	TP = np.trace(cm) # a.k.a 'hits'
	# FN a.k.a 'misses'
	total = np.sum(cm)
	FN = total - TP
	return TP, FN

# Plots
# Confusion matrix
def plotConfusionMatrix(cm, y_true):
	norm_conf = []
	for i in cm:
		a = 0
		tmp_arr = []
		a = sum(i, 0)
		for j in i:
			tmp_arr.append(float(j) / float(a))
		norm_conf.append(tmp_arr)

	fig = plt.figure(figsize=(8, 6))
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

	# cb = fig.colorbar(res)
	classes = sorted(set(y_true))
	plt.xticks(range(width), classes, rotation='vertical')
	plt.yticks(range(height), classes)
	plt.show()

# Cross-validation plots
# We have at least 5 parameters to optimise:
# C, gamma (SVM), scale, step (MS Dense SIFT)
# and number of clusters (BoVW)
#   - One 3D plot of accuracy vs C, gamma
#       (other params fixed)

# Utility function to move the midpoint of a colormap to be around
# the values of interest.

class MidpointNormalize(Normalize):

	def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
		self.midpoint = midpoint
		Normalize.__init__(self, vmin, vmax, clip)

	def __call__(self, value, clip=None):
		x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
		return np.ma.masked_array(np.interp(value, x, y))


def CVPlot_SVM(CV_grid, clf_params):
	print "The best estimator is:\n " + str(CV_grid.best_estimator_)
	print "\n"
	print "Obtained an accuracy of: %0.3f%% \n with clf_params: %s" \
		  % (100*CV_grid.best_score_, CV_grid.best_params_)

	print "Generating graphs for kernel_type: " + clf_params['kernel'][0]

	if clf_params['kernel'][0] == 'precomputed':
		# Do a 2D graph with changes only with respect to C (no gamma)
		C_range = clf_params['C']
		scores = CV_grid.cv_results_['mean_test_score'].reshape(len(C_range))
		title_str = "Best acccuracy: " + str(np.round(CV_grid.best_score_,4)) +\
					" for C=" + str(np.round(CV_grid.best_params_['C'], 4))
		# Make a simple figure
		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.plot(C_range, scores, '-o')
		ax.set_xscale('log')
		ax.set_title(title_str)
		ax.set_xlabel("C")
		ax.set_ylabel("Accuracy")
		ax.grid(True)

		# Add red horizontal line to highlight maximum accuracy
		#ax.plot(C_range, np.repeat(CV_grid.best_score_, len(C_range)))
		ax.axhline(CV_grid.best_score_, color='r', linestyle='--')
		x_min, _ = ax.get_xlim()
		_, y_max = ax.get_ylim()
		ax.text(1.1*x_min, 0.9*y_max, str(np.round(CV_grid.best_score_, 4)), color='r')
		plt.show()

	else: # Suppose 'rbf'==> C + gamma == heat map
		C_range = clf_params['C']
		gamma_range = clf_params['gamma']

		# Plot scores and parameters settings (C and gamma)
		scores = CV_grid.cv_results_['mean_test_score'].reshape(len(C_range),
															 len(gamma_range))

		# Make a nice figure

		plt.figure(figsize=(8, 6))
		plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
		plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
				   norm=MidpointNormalize(vmin=0.2, midpoint=0.72))
		title_str = "Best accuracy:" + str(np.round(CV_grid.best_score_, 4)) + \
					" for C=" + str(np.round(CV_grid.best_params_['C'], 4)) + \
					",gamma=" + str(np.round(CV_grid.best_params_['gamma'], 4))

		plt.xlabel('gamma')
		plt.ylabel('C')
		plt.title(title_str)
		plt.colorbar()
		plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
		plt.yticks(np.arange(len(C_range)), C_range)

		plt.show()

# For both plots below we have to save the model results or simply
# pass in a vector of scores.
#   - One 3D plot of accuracy vs scale and step
#       ( " " " )
#   - One 2D plot of accuracy vs num_centroids
#       ( " " " )


def printCV_resultSummary(CV_grid, feat_type):
	## ERASE THIS: THE GRAPHIC IS MORE THAN ENOUGH
	# 1st: print summary to screen
	print "Printing best CV results..."
	print "Feat. type: {!s}; Clf. type: SVM".format(feat_type)

	print("Best parameters set found on development set:")
	print("")
	print(CV_grid.best_params_)
	print("")
	print("Grid scores on development set (ordered by decreasing accuracy):")
	print("")
	means = CV_grid.cv_results_['mean_test_score']
	stds = CV_grid.cv_results_['std_test_score']
	params = CV_grid.cv_results_['params']

	# Sort by decreasing accuracy:
	index_sort = np.argsort(means)
	index_sort = index_sort[::-1]
	means_sorted = means[index_sort]
	stds_sorted = stds[index_sort]
	params_sorted = params[index_sort]

	i = 1
	for mean, std, param in zip(means_sorted, stds_sorted, params_sorted):
		print "Fit's result number: " + str(i) + " . Score + params."
		print("Accuracy=%0.3f (2*std=+/-%0.03f) for %r"
			  % (mean, std * 2, param))
		print("")
		i += 1