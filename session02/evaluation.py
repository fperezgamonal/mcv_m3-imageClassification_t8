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
#   - One 3D plot of accuracy vs scale and step
#       ( " " " )
#   - One 2D plot of accuracy vs num_centroids
#       ( " " " )