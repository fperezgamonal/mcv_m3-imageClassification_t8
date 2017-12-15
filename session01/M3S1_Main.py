import numpy as np
import cPickle
import time
from M3S1_Evaluation import M3S1_Evaluation
from M3S1_ImageFeatureExtractor import ImageFeatureExtractor
from M3S1_Classifier import Classifier

# read train and test files
def readInput(type):   
    try:
        if type == 'train':
            #train_image_filenames
            filenames = cPickle.load(open('train_images_filenames.dat','r')) 
            #train_labels
            labels = cPickle.load(open('train_labels.dat','r'))
            print 'Loaded '+str(len(filenames))+' training images filenames with classes ',set(labels) 
        elif type == 'test':
            #test_images_filenames
            filenames = cPickle.load(open('test_images_filenames.dat','r')) 
            #test_labels
            labels = cPickle.load(open('test_labels.dat','r'))
            print 'Loaded '+str(len(filenames))+' testing images filenames with classes ',set(labels)
    except ValueError:
        filenames = []
        labels = []
        print 'Nothing was loaded'
    return filenames, labels



# Extract SIFT features from an image
def extractAllFeatures(filenames, labels, featureExtractor):
    Train_descriptors = []
    Train_label_per_descriptor = []
    for i in range(len(filenames)):
        filename=filenames[i]
        print '===' + filename + '==='
        if Train_label_per_descriptor.count(labels[i])<30:
            Train_label_per_descriptor.append(labels[i])
            
            print 'Reading image '+filename
            keypoints, descriptors = featureExtractor.extractFeatures(filename)
            Train_descriptors.append(descriptors)
            
    D=Train_descriptors[0]
    L=np.array([Train_label_per_descriptor[0]]*Train_descriptors[0].shape[0])
    
    for i in range(1,len(Train_descriptors)):
		D=np.vstack((D,Train_descriptors[i]))
		L=np.hstack((L,np.array([Train_label_per_descriptor[i]]*Train_descriptors[i].shape[0])))        
            
    return D, L

#Testing k-NN classifier
def predictClass(filename, featureExtractor, classifier):  
    kpt,des = featureExtractor.extractFeatures(filename)
    predictions = classifier.predict(des)
    return_counts=True
	
    # Now we need to aggregate them all into a single image classification
    values, counts = np.unique(predictions, return_counts)
    predictedclass = values[np.argmax(counts)]
    
    return predictedclass



start = time.time()


    # read train and test files
train_image_filenames, train_labels = readInput('train')
test_images_filenames, test_labels = readInput('test')

featureExtractor = ImageFeatureExtractor()
featureExtractor.configureSIFT(100)

# Extract SIFT features from an image
D, L = extractAllFeatures(train_image_filenames, train_labels, featureExtractor)

# k-NN classifier
numNeighbors=5
classifier = Classifier()
classifier.configureKNN(numNeighbors, -1)
classifier.train(D, L)


# =======================
# Test with performance evaluation
# Get all test data and predict  their labels
print '==='
numtestimages=0
numcorrect=0
predictedclassList=[];
for i in range(len(test_images_filenames)):
    filename=test_images_filenames[i]
    #Testing k-NN classifier
    predictedclass = predictClass(test_images_filenames[i], featureExtractor, classifier)
    predictedclassList.append(predictedclass)
    
    print 'image '+filename+' was from class '+test_labels[i]+' and was predicted '+predictedclass
    numtestimages+=1
    if predictedclass==test_labels[i]:
        numcorrect+=1


# M3S1_Evaluation s'ha de passar "test_labels" i "predictedclassList"
eval = M3S1_Evaluation(predictedclassList);
eval.printEvaluation()

end=time.time()

ret =  'Done in '+str(end-start)+' secs.'
print ret

#return ret

## 30.48% in 302 secs.