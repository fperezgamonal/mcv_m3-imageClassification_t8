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
from features import computeTraining_descriptors, reduceDimensionality_PCA
from BoVW import computeCodebook, getBoVW_train, getBoVW_test
from classifier import clf_train, clf_predict
from sklearn.metrics import accuracy_score

# Define general variables and select the scheme to run with 'num':
num_scheme = 0
# Codebook
n_clusters = 512
# SVM
cost = 1
gma = 0.002
krnl = 'rbf'
# Evaluation
plotGraph = True
PCA_on = True


def run_scheme(scheme, num_clusters, C, gamma, kernel, plotGraphs, PCA):
    print "Running scheme with the following parameters: "
    print "Scheme num: " + str(scheme) + ", BoVW: num_clusters=" + str(num_clusters) +\
        "; SVM: C=" + str(C) + ", gamma=" + str(gamma) + ", kernel='" + kernel + "'" +\
        "; plotGraphs=" + str(plotGraphs) + "; PCA_on=" + str(PCA)
    # Initialise variables to 'scheme'-specific values
    if scheme == 0:
        descriptor_type = "SIFT"
        descriptor_param = np.array([300])
    elif scheme == 1:
        descriptor_type = "Dense_SIFT"
        descriptor_param = np.array([16]) # this is the step in pixels in between two SIFT descriptors
    elif scheme == 2:
        descriptor_type = "FisherVectors"
        descriptor_param = np.array([8])  # the number of GMMs used

    if PCA:
        num_cols = 64
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
    descriptors_np, Train_descriptors, kpt_dense = computeTraining_descriptors(descriptor_type, descriptor_param,
                                    train_images_filenames, train_labels, PCA, num_cols)

    # 3) Reduce number of features by PCA (reducing m=128 cols)
    # <<< DONE INTERNALLY TO AVOID PROBLEMS WHEN LOADING FROM DISK >>>
    # descriptors_np = reduceDimensionality_PCA(descriptors_np, num_cols)
    # ############ CHECK TRAIN_DESCRIPTORS DIMENSION, TO PROPERLY APPLY PCA
    # i = 0
    # for mtx in Train_descriptors:
    #     # Apply PCA to reduce dimension
    #     Train_descriptors[i] = reduceDimensionality_PCA(mtx, num_cols)
    #     i += 1

    # Store features (after PCA)
    # cPickle.dump([descriptors_np, Train_descriptors], open(D_filename, "wb"))

    # 4) Compute codebook
    codebook = computeCodebook(num_clusters, descriptors_np, descriptor_type,
                               descriptor_param)
    # 5) Get training BoVW
    train_VW = getBoVW_train(codebook, num_clusters, Train_descriptors)

    # 6) Train SVM
    clf, train_scaler = clf_train(train_VW, train_labels, C, gamma, kernel)

    # 7) Get test BoVW
    test_VW = getBoVW_test(codebook, num_clusters, test_images_filenames,
                           descriptor_type, descriptor_param, kpt_dense, PCA, num_cols)

    # 8) Get evaluation (accuracy, f-score, graphs, etc.)
    predictions = clf_predict(clf, train_scaler, test_VW, test_labels)
    # Get metrics and graphs:
    # We need to implement our own for latter integration with the rest of the project
    # Accuracy, F-score (multi-class=> average? add up?)

    # Momentarily (just to check that the separated scripts do the same as the original one)
    accuracy = accuracy_score(test_labels, predictions)
    print "Achieved " + str(100*accuracy) + "%"

    print "Printing metrics and creating plots..."


    end=time.time()
    print 'Everything done in '+str(end-start)+' secs.'
    ### 69.02% (default script (BoVW with SIFT + SVM))

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
	run_scheme(num_scheme, n_clusters, cost, gma, krnl, plotGraph, PCA_on)