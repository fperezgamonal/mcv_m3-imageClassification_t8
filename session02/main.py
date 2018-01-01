# TODO:
#   - Link all functions
#   - Define options (e.g.: SIFT + SVM, DenseSIFT + SVM, FisherVectors + SVM,etc.)
# Containing steps:
#   1) Load train and test data and labels
#   2) Extract features for the training data
#   3) Compute codebook
#   4) Get training BoVW
#   5) Train SVM
#   6) Get test BoVW
#   7) Get evaluation (now only accuracy is printed but we should include graphics
#      and more measures
import cPickle
import numpy as np
import time
from features import computeTraining_descriptors
from BoVW import computeCodebook, getBoVW_train, getBoVW_test
from classifier import clf_train, clf_predict
from evaluation import accuracy, precision, recall,\
    f1score, confusionMatrix, plotConfusionMatrix, HitsAndMisses

# Define general variables and select the scheme to run with 'num':
num_scheme = 1
# Initialise variables to 'scheme'-specific values
if num_scheme == 0:
    D_type = "SIFT"
    D_param = np.array([300])
elif num_scheme == 1:
    D_type = "Dense_SIFT"
    D_param = np.array([16]) # this is the step in pixels in between two SIFT descriptors
elif num_scheme == 2:
    D_type = "FisherVectors"
    D_param = np.array([8])  # the number of GMMs used
else:
    D_type = "SIFT"
    D_param = np.array([300])

# Codebook
n_clusters = 512
# SVM
# Reference params were==> C=1, gm=0.002
# Best in CV for MS Dense SIFT ==> C=10, gm=0.001
# Best in CV for SIFT ==> C=XX, gm = X.XXX
cost = 10
gma = 0.001
krnl = 'rbf'
# Evaluation
plotGraph = True
PCA_on = True
if PCA_on:
    n_cols = 64
else:
    n_cols = 128

def run_scheme(scheme, descriptor_type, descriptor_param,
               num_clusters, C, gamma, kernel, plotGraphs, PCAon, num_cols):
    print "Running scheme with the following parameters: "
    print "Scheme num: " + str(scheme) + ", BoVW: num_clusters=" + str(num_clusters) +\
        "; SVM: C=" + str(C) + ", gamma=" + str(gamma) + ", kernel='" + kernel + "'" +\
        "; plotGraphs=" + str(plotGraphs) + "; PCA_on=" + str(PCAon)
    start = time.time()

    # 1) Read the train and test files
    train_images_filenames = cPickle.load(open('train_images_filenames.dat','r'))
    test_images_filenames = cPickle.load(open('test_images_filenames.dat','r'))
    train_labels = cPickle.load(open('train_labels.dat','r'))
    test_labels = cPickle.load(open('test_labels.dat','r'))
    print 'Loaded '+str(len(train_images_filenames))+' training images filenames\
     with classes ',set(train_labels)
    print 'Loaded '+str(len(test_images_filenames))+' testing images filenames\
     with classes ',set(test_labels)

    # 2) Extract features (train)
    D, Train_descriptors, kpt_dense, pca_train, sclr_train = computeTraining_descriptors(descriptor_type, descriptor_param,
                                    train_images_filenames, train_labels, PCAon, num_cols)

    # 3) Reduce number of features by PCA (reducing m=128 cols)
    #   Computed internally in computeTraining_descriptors()
    # 4) Compute codebook
    codebook = computeCodebook(num_clusters, D, descriptor_type,
                               descriptor_param, PCAon)
    # 5) Get training BoVW
    train_VW = getBoVW_train(codebook, num_clusters, Train_descriptors)

    # 6) Train SVM
    clf, train_scaler = clf_train(train_VW, train_labels, C, gamma, kernel)

    # 7) Get test BoVW
    test_VW = getBoVW_test(codebook, num_clusters, test_images_filenames,
                           descriptor_type, descriptor_param,
                           kpt_dense, PCAon, pca_train, sclr_train)

    # 8) Get evaluation (accuracy, f-score, graphs, etc.)
    predictions = clf_predict(clf, train_scaler, test_VW, test_labels)
    # Get metrics and graphs:
    # We need to implement our own for latter integration with the rest of the project
    # Accuracy, F-score (multi-class=> average? add up?)

    acc = accuracy(test_labels, predictions)
    prec = precision(test_labels, predictions)
    rec = recall(test_labels, predictions)
    f1sc = f1score(test_labels, predictions)
    cm = confusionMatrix(test_labels, predictions)
    hits, misses = HitsAndMisses(cm)
    print "Confusion matrix:\n"
    print(str(cm))
    print("\n")
    print "Results (metrics):\n" + "Accuracy= {:04.2f}%\n" \
                                   "Precision= {:04.2f}%\n" \
                                   "Recall= {:04.2f}%\n" \
                                   "F1-score= {:04.2f}%\n" \
                                   "Hits(TP)={:d}\n" \
                                   "Misses(FN)={:d}\n".format(
        100*acc, 100*prec, 100*rec, 100*f1sc, hits, misses)
    print("\n")
    if plotGraphs:
        # Plot confusion matrix (and any other graph)
        print "Plotting confusion matrix..."
        plotConfusionMatrix(cm, test_labels)

    end=time.time()
    print 'Everything done in '+str(end-start)+' secs.'
    ### 68.65% (default script (BoVW with SIFT + SVM)) C= 1, gm=.002
    ### 83.89% MS Dense SIFT (step=8 ==>huge model, ~9GBs of features)
    ### 78.07% MS Dense SIFT (smaller step, <700MBs of features)
    ### 82.00% MS Dense SIFT (step = 16) with optimal CV params(see above)
    ### 69.89% Same params as default but with PCA (64 cols)
    ### XX.XX% Same params as Optimal CV dense SIFT (82.00%)

if __name__ == "__main__":
	# main of this function
	print "Executing session02/main.py..."

# change 'num' value (top) to run routine number 'num'. Mappings:
#	'num' value			routine
#	  0					(BoVW) SIFT + SVM
#	  1					(BoVW) Dense SIFT + SVM
#     2					(BoVW) Fisher Vectors + SVM
#	  N                 (BoVW) routineXX

# Execute 'num' scheme ('num' is defined at the top)
	run_scheme(num_scheme, D_type, D_param, n_clusters,
               cost, gma, krnl, plotGraph, PCA_on, n_cols)