# TODO:
# Include all evaluation-related functions, which are:
#   - For now use scikit's: confusion matrix, accuracy, fscore, etc. (later create our OWN)
#   - print and plot results of cross validation and 'standard' execution (main.py)
#       * One function for cross validation ('cross_val.py'), one for 'main.py'
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
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

def CVPlot_SVM(CV_grid, clf_params):
    print "The best classifier is: " + CV_grid.best_estimator_

    # Mappings from clf_params to actual params
    # (in case we want to add more)
    C_range = clf_params['C']
    gamma_range = clf_params['gamma']
    # kernel_type = clf_params['kernel']

    # Plot scores and parameters settings (C and gamma)
    score_dict = CV_grid.grid_scores_

    # We extract just the scores
    scores = CV_grid.cv_results_['mean_test_score'].reshape(len(C_range),
                                                         len(gamma_range))

    # Make a nice figure

    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot)
    plt.xlabel('gamma')
    plt.ylabel('C')
    plt.title("Cross-validation accuracy as a function of (C, gamma)")
    plt.colorbar()
    plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
    plt.yticks(np.arange(len(C_range)), C_range)
    plt.title('Validation accuracy')
    plt.show()

# For both plots below we have to save the model results or simply
# pass in a vector of scores.
#   - One 3D plot of accuracy vs scale and step
#       ( " " " )
#   - One 2D plot of accuracy vs num_centroids
#       ( " " " )


def printCV_resultSummary(CV_grid, feat_type):
	# 1st: print summary to screen
	print "Printing best CV results..."
	print "Feat. type: {!s}; Clf. type: SVM".format(feat_type)

	print("Best parameters set found on development set:")
	print()
	print(CV_grid.best_params_)
	print()
	print("Grid scores on development set:")
	print()
	means = CV_grid.cv_results_['mean_test_score']
	stds = CV_grid.cv_results_['std_test_score']
	for mean, std, params in zip(means, stds, CV_grid.cv_results_['params']):
		print("%0.3f (+/-%0.03f) for %r"
			  % (mean, std * 2, params))
	print()