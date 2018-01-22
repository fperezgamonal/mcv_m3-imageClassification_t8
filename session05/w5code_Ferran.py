#coding=utf-8
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Sequential, Flatten, Dense
from keras.layers import Conv2D, GlobalAveragePooling2D, Dropout, BatchNormalization
from keras.initializers import Constant
from keras.optimizers import SGD, Adam, Adadelta, Adagrad
from keras import backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau
from keras.utils.vis_utils import plot_model
#from keras.utils.visualize_util import plot (use this if
# the import above does not work). This is from an older Keras version.
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import time
# GPU-server specific
# imports
import os
import getpass

# Use only our gpu
os.environ["CUDA_VISIBLE_DEVICES"]=getpass.getuser()[-1]

# Dataset location
train_data_dir = '/share/datasets/MIT_split/test'
val_data_dir = '/share/datasets/MIT_split/test'
test_data_dir = '/share/datasets/MIT_split/test'

# Original S04 script path (different folder structure)
# train_data_dir='/data/MIT/train'
# val_data_dir='/data/MIT/test'
# test_data_dir='/data/MIT/test'
# The dataset locally is here (like last week):
#train_data_dir = '../../Databases/MIT_split/train'
#val_data_dir = '../../Databases/MIT_split/test'
#test_data_dir = '../../Databases/MIT_split/test'

# Basic parameters
img_width = 256
img_height=256
batch_size=32
number_of_epoch=50

plot = False  			# Whether to compute acc and loss curves or not
# Note: since we are using TensorBoard(LIVE results), we can use its curves and
# access to more details instead of those plots. Additionaly, we have run into
# problems while saving these graphs in the server.

dropout = False
batch_norm = False
drop_prob = 0.5
n_examples = 1881		# Number of training examples used (def=1881)
# If n_examples > 1881, data augmentation is used


# Model filename
model_fname = "modelCNN-n_examples=" + str(n_examples) +\
		"_blocks_used=" + str(blocks_used) + "_dropout=" +\
		str(dropout) + "_p=" + str(drop_prob) + "_batchNorm=" +\
		str(batch_norm) + ".h5"

# Pre-processing (zero-center images)
def preprocess_input(x, dim_ordering='default'):
	if dim_ordering == 'default':
		dim_ordering = K.image_dim_ordering()
	assert dim_ordering in {'tf', 'th'}

	if dim_ordering == 'th':
		# 'RGB'->'BGR'
		x = x[ ::-1, :, :]
		# Zero-center by mean pixel
		x[ 0, :, :] -= 103.939
		x[ 1, :, :] -= 116.779
		x[ 2, :, :] -= 123.68
	else:
		# 'RGB'->'BGR'
		x = x[:, :, ::-1]
		# Zero-center by mean pixel
		x[:, :, 0] -= 103.939
		x[:, :, 1] -= 116.779
		x[:, :, 2] -= 123.68
	return x

# Definition of the CNN architecture
model = Sequential()
model.add(Conv2D(32, (3,3), padding='same', activation='relu',
		input_shape(img_width, img_height, 3), name='conv_1'))
model.add(MaxPooling2D(pool_size=(2,2), name='pool_1'))

if batch_norm:
	model.add(BatchNormalization())

model.add(Conv2D(64, (3,3), padding='same', activation='relu', name='conv_2'))
model.add(MaxPooling2D(pool_size=(2,2), name='pool_2'))

model.add(Flatten())
model.add(Dense(2048, activation='relu', name='dense_1'))

if dropout:
	model.add(Dropout(drop_prob, name='drop_1'))
model.add(Dense(1024, activation='relu', name='dense_2'))

if dropout:
	model.add(Dropout(drop_prob, name='drop_2'))

model.add(Dense(8, activation='softmax', name='predictions'))


print("")
print("Model summary (layers (dimension+name) + num. params/layer):")
print("")
model.summary()
# Sleep to visualize the architecture that is about to be trained
time.sleep(3)
plot_model(model, to_file=model_fname.replace('.h5','.png',1), show_shapes=True, show_layer_names=True) 

if os.path.isfile(model_fname):
	print("This model was already trained, overwritting may occur...")
# With 'Adam', lr=0.001 is the default. Change according to the results!
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

# Define checkpoint to store the best model only
model_checkpoint = ModelCheckpoint(model_fname, monitor='val_acc', verbose=1,
save_best_only=True, mode='max')
# TensorBoard
# ---- Instructions ----
# When the modelling has started training, open a new terminal and enter:
# tensorboard --logdir ./TensorBoard_graphs (open the URL provided)
# Note: we give each model with a different fname is written to a
# different directory so we can compare among runs (configure
# filename properly when changing model params!)
tbCallback = TensorBoard(log_dir='./TensorBoard_graphs/' +\
	model_fname.replace('.h5','',1) + '/', histogram_freq=1,
				write_graph=True, write_images=True)

# Early stopping, to avoid having to select the nb_epochs as a parameter,
# configure early stopping and stop after X epochs of worse val_acc
earlyStopping = EarlyStopping(monitor='val_acc', patience=5, verbose=0, mode='max')

# Reduce learning rate on plateau
# Sometimes it helps to reduce the learning rate when the loss/accuracy does not improve
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=.5, patience=3,
				min_lr=1e-5)

# Add all callbacks to a list
callbacks_list = [model_checkpoint, tbCallback, earlyStopping, reduce_lr]

#preprocessing_function=preprocess_input,
if num_examples > 1881: # create more samples w. data augmentation
	datagen = ImageDataGenerator(featurewise_center=False,
        	samplewise_center=False,
        	featurewise_std_normalization=True,
        	samplewise_std_normalization=False,
        	preprocessing_function=preprocess_input,
        	rotation_range=10.,
        	width_shift_range=0.2,
        	height_shift_range=0.2,
        	shear_range=0.2,
        	zoom_range=.5,
        	channel_shift_range=0.,
        	fill_mode='reflect',
        	cval=0.,
        	horizontal_flip=True,
        	vertical_flip=False,
        	rescale=1./255)
else:
	datagen = ImageDataGenerator(featurewise_center=False,
		samplewise_center=False,
		featurewise_std_normalization=True,
		samplewise_std_normalization=False,
		preprocessing_function=preprocess_input,
		rotation_range=0.,
		width_shift_range=0.,
		height_shift_range=0.,
		shear_range=0.,
		zoom_range=0.,
		channel_shift_range=0.,
		fill_mode='nearest',
		cval=0.,
		horizontal_flip=False,
		vertical_flip=False,
		rescale=1./255)

train_generator = datagen.flow_from_directory(train_data_dir,
		target_size=(img_width, img_height),
		batch_size=batch_size,
		class_mode='categorical')
# Change to create an independent folder for test (??)
test_generator = datagen.flow_from_directory(test_data_dir,
		target_size=(img_width, img_height),
		batch_size=batch_size,
		class_mode='categorical')

validation_generator = datagen.flow_from_directory(val_data_dir,
		target_size=(img_width, img_height),
		batch_size=batch_size,
		class_mode='categorical')

history=model.fit_generator(train_generator,
		steps_per_epoch=int(n_examples//batch_size +1),# number of batches per epoch
		epochs=number_of_epoch,
		validation_data=validation_generator,
		validation_steps=807,			# if test has a new folder this is reduced! 
		callbacks=callbacks_list)

# Change this if need be (see above)
result = model.evaluate_generator(test_generator, steps=807)
print("Test results (Loss + Accuracy):" + str(result)) 


# list all data in history

if plot:
  # summarize history for accuracy
  plt.plot(history.history['acc'])
  plt.plot(history.history['val_acc'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'validation'], loc='upper left')
  plt.savefig('accuracy.jpg')
  plt.close()
  # summarize history for loss
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'validation'], loc='upper left')
  plt.savefig('loss.jpg')
