# Based on main.py(session02) and cross_val.py from session01 but simplified
# (as we only use SVM as classifier)
import cPickle
import time
from features import computeTraining_descriptors
from BoVW import computeCodebook, getBoVW_train, getBoVW_test
from classifier import clf_train, clf_predict
from sklearn.metrics import accuracy_score

# Define general variables and select the scheme to run with 'num':
num_scheme = 1
# Codebook
num_clusters = 512
# SVM
C = 1
gamma = 0.002
kernel = 'rbf'
# Evaluation
plotGraphs = True


def run_scheme(scheme, num_clusters, C, gamma, kernel, plotGraphs):
	print "Running scheme with the following parameters: "
	print "Scheme num: " + str(scheme) + ", BoVW: num_clusters=" + str(num_clusters) +\
		"; SVM: C=" + str(C) + ", gamma=" + str(gamma) + ", kernel='" + kernel + "'" +\
		"; plotGraphs=" + str(plotGraphs)
	# Initialise variables to 'scheme'-specific values
	if scheme == 0:
		descriptor_type = "SIFT"
		descriptor_param = [50, 100, 150]
	elif scheme == 1:
		descriptor_type = "Dense_SIFT"
        # step already tested 10 (super-slow)
		descriptor_param = [16, 32, 64] # this is the step in pixels in between two SIFT descriptors
	elif scheme == 2:
		descriptor_type = "FisherVectors"
		descriptor_param = 8  # the number of GMMs used

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
	descriptors_np, Train_descriptors = computeTraining_descriptors(descriptor_type, descriptor_param,
									train_images_filenames, train_labels)
	# 3) Compute codebook
	codebook = computeCodebook(num_clusters, descriptors_np, descriptor_type,
							   descriptor_param)
	# 4) Get training BoVW
	train_VW = getBoVW_train(codebook, num_clusters, Train_descriptors)

	# 5) Train SVM
	clf, train_scaler = clf_train(train_VW, train_labels, C, gamma, kernel)

	# 6) Get test BoVW
	test_VW = getBoVW_test(codebook, num_clusters, test_images_filenames,
						   descriptor_type, descriptor_param)

	# 7) Get evaluation (accuracy, f-score, graphs, etc.)
	predictions = clf_predict(clf, train_scaler, test_VW, test_labels)
	# Get metrics and graphs:
	# We need to implement our own for latter integration with the rest of the project
	# Accuracy, F-score (multi-class=> average? add up?)

	# Momentarily (just to check that the separated scripts do the same as the original one)
	accuracy = accuracy_score(test_labels, predictions)
	print "Achieved " + str(100*accuracy) + "%"

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
	run_scheme(num_scheme, num_clusters, C, gamma, kernel, plotGraphs)

def plotCV_results(CV_grid, feat_type, clf_type):
	# 1st: print summary to screen
	print "Printing best CV results..."
	print "Feat. type: {!s}; Clf. type: {!s}".format(feat_type,clf_type)

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

	# 2nd: generate graph with the results
	print "Plotting CV results...(to be implemented)"