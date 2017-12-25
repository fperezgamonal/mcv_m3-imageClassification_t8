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
from sklearn.preprocessing import StandardScaler
from sklearn import svm

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