from utils import *
from keras.models import Sequential
from keras.layers import Flatten, Dense, Reshape, Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.constraints import maxnorm
from keras.callbacks import ModelCheckpoint

from keras import regularizers
from keras.initializers import Constant, glorot_normal
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
import matplotlib.pyplot as plt
# GPU-server specific imports
import os
import getpass

#os.environ["CUDA_VISIBLE_DEVICES"]=getpass.getuser()[-1]

#user defined variables
IMG_SIZE    = 32
BATCH_SIZE  = 16
DATASET_DIR = '../../Databases/MIT_split'
#DATASET_DIR = '/share/datasets/MIT_split'
NUM_HIDDEN = 4
DIM_HIDDEN = [4096, 1024, 4096, 256]# [512,128]#[2048, 1024, 1024]
L2_reg = True
dropout_reg = True
drop_prob = 0.5
MODEL_FNAME = "MLP_numHidden=" + str(NUM_HIDDEN) + "_dim=" +\
			str(DIM_HIDDEN) + "_regL2=" + str(L2_reg) +\
			  "_regDropout=" + str(dropout_reg)

if not os.path.exists(DATASET_DIR):
  colorprint(Color.RED, 'ERROR: dataset directory '+DATASET_DIR+' do not exists!\n')
  quit()

if os.path.isfile(MODEL_FNAME + '.h5'):
	colorprint(Color.YELLOW, 'WARNING: this model was already built before\n')
	quit()

tmp_str = "Building MLP model with" + str(NUM_HIDDEN) +\
		  " hidden layers and dimensions=" + str(DIM_HIDDEN) +\
		  ". L2reg=" + str(L2_reg) + ", dropoutReg=" +\
		  str(dropout_reg)
colorprint(Color.BLUE, tmp_str)

#Build the Multi Layer Perceptron model
model = Sequential()
model.add(Reshape((IMG_SIZE*IMG_SIZE*3,),input_shape=(IMG_SIZE, IMG_SIZE, 3),name='reshape'))
#model.add(Dense(units=2048, activation='relu',name='second'))
#model.add(Dropout(0.2))
model.add(Dense(units=DIM_HIDDEN[0], activation='relu',
				kernel_initializer='glorot_normal',
				bias_initializer=Constant(value=0.01),
				name='first_dense',
				kernel_constraint=maxnorm(5),
				kernel_regularizer=regularizers.l2(0.01)))
				#kernel_constraint=maxnorm(3)))	
				#kernel_constraint=maxnorm(3)))
#model.add(BatchNormalization())				
#kernel_regularizer=regularizers.l2(0.001)))
				#kernel_constraint=maxnorm(3)))
# kernel_regularizer=regularizers.l2(0.001)))
model.add(Dropout(drop_prob))

model.add(Dense(units=DIM_HIDDEN[1], activation='relu',
				kernel_initializer='glorot_normal',
				bias_initializer=Constant(value=0.01),
				name='second_dense',
				kernel_regularizer=regularizers.l2(0.01),
				kernel_constraint=maxnorm(5)))#, kernel_constraint=maxnorm(3)))

#model.add(BatchNormalization())
				#kernel_regularizer=regularizers.l2(0.001)))
				#kernel_constraint=maxnorm(3)))
# kernel_regularizer=regularizers.l2(0.001)))
model.add(Dropout(drop_prob))

model.add(Dense(units=DIM_HIDDEN[2], activation='relu',
				kernel_initializer='glorot_normal',
				bias_initializer=Constant(value=0.01),
				name='third_dense',
				kernel_regularizer=regularizers.l2(0.01),
				kernel_constraint=maxnorm(5)))#, kernel_constraint=maxnorm(3)))

# REGULARIZATION AND NORMALIZATION
#model.add(BatchNormalization())				
#kernel_regularizer=regularizers.l2(0.001)))
				#kernel_constraint=maxnorm(3)))
#kernel_regularizer=regularizers.l2(0.001)))
model.add(Dropout(drop_prob))

model.add(Dense(units=DIM_HIDDEN[3], activation='relu',
				kernel_initializer='glorot_normal',
				bias_initializer=Constant(value=0.01),
				name='fourth_dense',
				kernel_regularizer=regularizers.l2(0.01),
				kernel_constraint=maxnorm(5)))
model.add(Dropout(drop_prob))
#model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=8, activation='softmax', name='output'))

sgd = SGD(lr=0.001, momentum=0.9, decay=1.0e-6, nesterov=True)
model.compile(loss='categorical_crossentropy',
			  optimizer=sgd,
			  metrics=['accuracy'])
filepath = MODEL_FNAME + '.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1,
				save_best_only=True, mode='max')
callbacks_list = [checkpoint]

print(model.summary())
plot_model(model, to_file='model' + MODEL_FNAME + '.png',
		   show_shapes=True, show_layer_names=True)

colorprint(Color.BLUE, 'Done!\n')

if os.path.exists(MODEL_FNAME):
  colorprint(Color.YELLOW, 'WARNING: model file '+MODEL_FNAME+ '.h5 exists and will be overwritten!\n')

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
		class_mode='categorical')  # since we use binary_crossentropy loss, we need categorical labels

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
		DATASET_DIR+'/test',
		target_size=(IMG_SIZE, IMG_SIZE),
		batch_size=BATCH_SIZE,
		classes = ['coast','forest','highway','inside_city','mountain','Opencountry','street','tallbuilding'],
		class_mode='categorical')

history = model.fit_generator(
		train_generator,
		steps_per_epoch=1881 // BATCH_SIZE,
		epochs=400,
		validation_data=validation_generator,
		validation_steps=807 // BATCH_SIZE,
		callbacks=callbacks_list)

colorprint(Color.BLUE, 'Done!\n')
colorprint(Color.BLUE, 'Saving the model into '+MODEL_FNAME+'.h5 \n')
#model.save_weights(MODEL_FNAME+'.h5')  # always save your weights after training or during training
colorprint(Color.BLUE, 'Done!\n')

  # summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('accuracy' + MODEL_FNAME + '.jpg')
plt.close()
  # summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('loss' + MODEL_FNAME + '.jpg')
