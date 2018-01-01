# classifier.py (SVM)
# TODO:
# Functions to include:
#   - Train (include training normalisation, store it and then train)
#   - Predict ('load' training normalisation, applied it and predict)
#
# Parameters necessary for later integration with CV
#   - Add at least the following SVM' train params as input variables:
#       * 'C', 'gamma', 'kernel'

# FOR NOW: copied code related to features (so we can see in red needed imports,etc.)
import time
import sys
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

# Train an SVM classifier
def clf_train(train_VW, train_labels, C, gamma, kernel):
    print "Training the SVM classifier..."
    initClf = time.time()
    stdSlr = StandardScaler().fit(train_VW)
    D_scaled = stdSlr.transform(train_VW)
    # Default was kernel 'rbf', C=1 and gamma .002
    clf = svm.SVC(kernel=kernel, C=C, gamma=gamma)
    clf.fit(D_scaled, train_labels)
    endClf = time.time()
    print "Training done in " + str(endClf-initClf) + " secs."
    return clf, stdSlr

# Test the classification accuracy (predict)
def clf_predict(clf, train_scaler, VW_test, test_labels):
    print "Testing the SVM classifier..."
    init_p=time.time()
    # We are not only interested in accuracy, we want the actual predictions
    # to compute OUR OWN accuracy, f-score, etc...
    predictions = clf.predict(train_scaler.transform(VW_test))
    end_p=time.time()
    print "Predictions computed in " + str(end_p-init_p) + " secs."
    return predictions

# Cross-validate training for different clf values (feat_values changed in cross_val.py)
def KFoldCrossValidation(train_VW, train_labels, n_folds, clf_params):
    # - n_folds: number of folds ('k')
    # - clf_params: dictionary with the classifier values to cross-validate ('C', 'gamma', etc.)

    # 1) Normalise training BoVW features (out of CV is done inside clf_train())
    #   * We need to return this normalisation as it will be needed for the 'test_set'
    print "Scaling features..."
    stdSlr = StandardScaler().fit(train_VW)
    D_scaled = stdSlr.transform(train_VW)
    print "Done scaling features"

    print "Initialising and performing K-fold Cross Validation..."
    init = time.time()
    # Initialise stratifiedKFold and GridSearchCV
    kfolds = StratifiedKFold(n_splits=n_folds, shuffle=False, random_state=42)
    grid = GridSearchCV(svm.SVC(), param_grid=clf_params, cv=kfolds, scoring='accuracy',
                        error_score=0, n_jobs=-1, verbose=10)

    # Start fitting all the combinations (N fits)
    grid.fit(D_scaled, train_labels)

    end = time.time()
    print "Finished K-fold Cross-validation. Done in " + str(end-init) + " secs."
    return stdSlr, grid

# Histogram intersection kernel (min)
# Used to create gram(kernel) matrix and work with test data
def histogramIntersectionKernelGramMatrix(M, N):
    # Implementation of the "Histogram Intersection Kernel"
    # K_int(M, N) = sum_{i=1}^m min{a_i, b_i}
    # with a_i, b_i >= 0
    # M, N histogram of images M_im and N_im
    # m is the number of bins of M and N
    kernel_matrix = np.zeros((M.shape[0], N.shape[0]))

    for i, m in enumerate(M):
        for j, n in enumerate(N):
            m = m.flatten()
            n = n.flatten()
            kernel_matrix[i,j] = np.sum(np.minimum(m, n))

    return kernel_matrix