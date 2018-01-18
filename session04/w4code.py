# coding=utf-8
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Flatten
from keras.layers import Dense, GlobalAveragePooling2D
from keras.initializers import Constant
from keras.optimizers import SGD, Adam, Adadelta, Adagrad
from keras import backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
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
train_data_dir = '/share/datasets/MIT_split/train'
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
img_width = 224
img_height=224
batch_size=32
number_of_epoch=50
train_from_scratch = False# True #False 	# train all the layers (including whole VGG16)

plot = False  			# Whether to compute acc and loss curves or not
# Note: since we are using TensorBoard(LIVE results), we can use its curves and
# access to more details instead of those plots. Additionaly, we have run into
# problems while saving these graphs in the server.

dropout = False
drop_prob = 0.5
blocks_used = 4			# blocks of the original VGG net used (def=5)
n_examples = 1881		# Number of training examples used (def=1881)

# Task-specific model weights (topModel==> added layers)
top_fc_fname = "top_FC_layers-n_examples=" + str(n_examples) +\
		"_blocks_used=" + str(blocks_used) + "_dropout=" +\
		str(dropout) + "_p=" + str(drop_prob) + ".h5"

# Check if we are training (and can) the whole model or just training new layers
if (train_from_scratch == False): # leave model_fname as is
	print("We will train ONLY the new layers added to the base model")
	model_fname = top_fc_fname
else:
	model_fname = top_fc_fname.replace("top_FC_layers", "whole_model", 1)

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

base_model = VGG16(weights='imagenet')
if not os.path.isfile('modelVGG16a.png'):
	plot_model(base_model, to_file='modelVGG16a.png', show_shapes=True, show_layer_names=True)

# DEBUGGING ONLY
#print("")
#print("Base model (VGG16) summary:")
#print("")
#base_model.summary()

# Layers specific to our classification task
# Cases depening on the number of blocks used
if blocks_used == 3: # only first 3 blocks (EXTRA test if we have time!)
	# VERY DEMANDING IN MEMORY RESOURCES
	#x = base_model.layers[-13].output (more clear like the line below)
	x = base_model.get_layer("block3_pool").output
        # Flatten block3_conv4+pool' output
        x = Flatten()(x)
        x = Dense(1024, activation='relu')(x)
	x = Dense(512, activation='relu')(x)
#,
		#kernel_initializer='glorot_normal',
		#bias_initializer=Constant(value=0.1),
		#name='own_first_FC')(x)
        #x = Dense(4096, activation='relu')(x)

elif blocks_used == 4: # up to block 4 
	#x = base_model.layers[-9].output
	x = base_model.get_layer("block4_pool").output
	x = Flatten()(x)
	# Add FC layers (1 for now, 2 uses too much memory in theory)
	# Flatten block4_conv4+pool' output
	#x = Flatten()(x)
	x = Dense(1024, activation='relu')(x)
	x = Dense(1024, activation='relu')(x)
	#,
                #kernel_initializer='glorot_normal',
                #bias_initializer=Constant(value=0.1),
                #name='own_first_FC')(x)
	#x = Dense(4096, activation='relu')(x)

else:	# complete VGG is used
	# Simply get last FC output and change the classifier (done once outside 'if-else')
	#x = base_model.layers[-2].output	
	x = base_model.get_layer("fc2").output

predictions = Dense(8, activation='softmax', name='predictions')(x)

model = Model(inputs=base_model.input, outputs=predictions)

print("")
print("Complete model (VGG16 + modifications) summary:")
print("")
model.summary()
# Sleep to visualize the architecture that is about to be trained
time.sleep(5)
plot_model(model, to_file='modelVGG16b_up2block=' + str(blocks_used) + '.png', show_shapes=True, show_layer_names=True)

if not train_from_scratch:
	for layer in base_model.layers:
		layer.trainable = False
else: # if train_from_scratch
	# Load top_model (FC layers) weights:
	model.load_weights(top_fc_fname)

model.compile(loss='categorical_crossentropy', optimizer=Adadelta(lr=0.01), metrics=['accuracy'])
for layer in model.layers:
	print(layer.name, layer.trainable)

# Define checkpoint to store the best model only
model_checkpoint = ModelCheckpoint(model_fname, monitor='val_acc', verbose=1,
			save_best_only=True, mode='max')
# Include TensorBoard here!
# When the modelling has started training, open a new terminal and enter:
# tensorboard --logdir ./TensorBoard_graphs (open the URL provided)
# Note: we give each model with a different fname is written to a
# different directory so we can compare among runs (configure
# filename properly when changing model params!)
tbCallback = TensorBoard(log_dir='./TensorBoard_graphs/' +\
	model_fname.replace('.h5','',1), histogram_freq=1,
				write_graph=True, write_images=True)

# Early stopping, to avoid having to select the nb_epochs as a parameter,
# configure early stopping and stop after X epochs of worse val_acc
earlyStopping = EarlyStopping(monitor='val_acc', patience=5, verbose=0, mode='max')
# Add all callbacks to a list
callbacks_list = [model_checkpoint, tbCallback, earlyStopping]

#preprocessing_function=preprocess_input,
datagen = ImageDataGenerator(featurewise_center=False,
	samplewise_center=False,
	featurewise_std_normalization=False,
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
	rescale=None)

train_generator = datagen.flow_from_directory(train_data_dir,
		target_size=(img_width, img_height),
		batch_size=batch_size,
		class_mode='categorical')

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
		validation_steps=807,
		callbacks=callbacks_list)


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
