from utils import *
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Reshape
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
import matplotlib.pyplot as plt
from keras import regularizers
from keras import optimizers
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import numpy as np

#user defined variables
IMG_SIZE    = 32
BATCH_SIZE  = 32
DATASET_DIR = '../../Databases/MIT_split'
MODEL_FNAME = 'mlp.h5'

if not os.path.exists(DATASET_DIR):
  colorprint(Color.RED, 'ERROR: dataset directory '+DATASET_DIR+' do not exists!\n')
  quit()


colorprint(Color.BLUE, 'Building MLP model...\n')

def create_model():
	#Build the Multi Layer Perceptron model
	model = Sequential()
	model.add(Reshape((IMG_SIZE*IMG_SIZE*3,),input_shape=(IMG_SIZE, IMG_SIZE, 3),name='first'))
	model.add(Dense(units=2048, activation='relu',name='second'))
	model.add(Dense(units=1024, activation='relu',name='third'))
	model.add(Dense(units=512, activation='relu',name='fourth'))
	model.add(Dense(units=256, activation='relu',name='fifth'))
	model.add(Dense(units=8, activation='softmax',name='sixth'))

	sgd = optimizers.SGD(lr=0.001, momentum=0.9, decay=1.0e-6, nesterov=False)
	model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

	print(model.summary())
	plot_model(model, to_file='modelMLP.png', show_shapes=True, show_layer_names=True)

	return model


colorprint(Color.BLUE, 'Done!\n')

if os.path.exists(MODEL_FNAME):
  colorprint(Color.YELLOW, 'WARNING: model file '+MODEL_FNAME+' exists and will be overwritten!\n')

colorprint(Color.BLUE, 'Start training...\n')

# this is the dataset configuration we will use for training
# only rescaling
train_datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True)

# this is the dataset configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        DATASET_DIR+'/train',  # this is the target directory
        target_size=(IMG_SIZE, IMG_SIZE),  # all images will be resized to IMG_SIZExIMG_SIZE
        batch_size=BATCH_SIZE,
        classes = ['coast','forest','highway','inside_city','mountain','Opencountry','street','tallbuilding'],
        shuffle=True,
        class_mode='categorical')  # since we use binary_crossentropy loss, we need categorical labels

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        DATASET_DIR+'/test',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        classes = ['coast','forest','highway','inside_city','mountain','Opencountry','street','tallbuilding'],
        shuffle=True,
        class_mode='categorical')

def getXYfromGenerator(generator):
	X,Y = generator.next()
	batch_index = 1
	
	while batch_index <= generator.batch_index:
		auxX, auxY = generator.next()
		X = np.concatenate((X,auxX))
		Y = np.concatenate((Y,auxY))
		batch_index = batch_index + 1

	return X,Y

#model = create_model()

''' It'll be the RandomSearch to fit
	without fit_generator
history = model.fit_generator(
        train_generator,
        steps_per_epoch=1881 // BATCH_SIZE,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=807 // BATCH_SIZE)
'''

X,Y = getXYfromGenerator(train_generator)
'''
history = model.fit(x=X, y=Y,
        #steps_per_epoch=1881 // BATCH_SIZE,
        epochs=50,
        validation_split=.3,
		batch_size=BATCH_SIZE,
		verbose=2)
'''

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV

modelClassifier = KerasClassifier(build_fn=create_model)

params = dict(epochs=np.arange(1,50))

random_search = RandomizedSearchCV(modelClassifier,
								   param_distributions=params,
								   n_iter=2,
								   n_jobs=1,
								   verbose=True)

results = random_search.fit(X,Y)

print("Best: %f using %s" % (results.best_score_, results.best_params_))
means = results.cv_results_['mean_test_score']
stds = results.cv_results_['std_test_score']
params = results.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))