# coding=utf-8
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Flatten
from keras.layers import Dense, GlobalAveragePooling2D
from keras.initializers import Constant
from keras import backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras.utils.vis_utils import plot_model
#from keras.utils.visualize_util import plot (use this if
# the import above does not work). This is from an older Keras version.
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
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
train_from_scratch = False # whether to train the WHOLE vgg with the changes
# or just the modified part (for S04 this should be set to False most of the times)

plot = True  # Whether to compute acc and loss curves or not
# Note: if we make TensorBoard(LIVE results) work, we can use its curves and
# access to more details instead of those plots.

# VGG base model filename (to load it and not download it,etc)
#baseModel_fname ="VGG16_allWeights.h5" # Should not be needed if keras detects that VGG weights have
# already been stored

dropout = False
drop_prob = 0.5
blocks_used = 4 # blocks of the original VGG net used (def=5)
n_examples = 1881 # Number of training examples used (def=1881)

# Task-specific model weights
model_fname = "scenes_FC_layers-n_examples=" + str(n_examples) +\
		"_blocks_used=" + str(blocks_used) + "_dropout=" +\
		str(dropout) + "_p=" + str(drop_prob) + ".h5"

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

#if os.path.isfile(baseModel_fname):
#	# Load base model
#	base_model = VGG16(weights='imagenet')#it can be loaded like this and it SHOULD NOT be downloaded again (loaded form memory instead)
#	#base_model.load_weights(baseModel_fname) # Load stored weights (from unique download)
#else:	
#	# create the base pre-trained model (done once)
#	base_model = VGG16(weights='imagenet')
#	plot_model(base_model, to_file='modelVGG16a.png', show_shapes=True, show_layer_names=True)
#	# Save model to avoid downloading for each execution
#	base_model.save_weights(baseModel_fname)

base_model = VGG16(weights='imagenet')
if not os.path.isfile('modelVGG16a.png'):
	plot_model(base_model, to_file='modelVGG16a.png', show_shapes=True, show_layer_names=True)

print("")
print("Base model (VGG16) summary:")
print("")
base_model.summary()

# Layers specific to our classification task
# Cases depening on the number of blocks used
if blocks_used == 3: # only first 3 blocks (EXTRA test if we have time!)
	x = base_model.layers[-13].output
        # Flatten block3_conv4+pool' output
        x = Flatten()(x)
        x = Dense(2048, activation='relu')(x)#,
		#kernel_initializer='glorot_normal',
		#bias_initializer=Constant(value=0.1),
		#name='own_first_FC')(x)
        #x = Dense(4096, activation='relu')(x)

elif blocks_used == 4: # up to block 4
	x = base_model.layers[-9].output
	# Add FC layers (1 for now, 2 uses too much memory in theory)
	# Flatten block4_conv4+pool' output
	x = Flatten()(x)
	x = Dense(2048, activation='relu')(x)#,
                #kernel_initializer='glorot_normal',
                #bias_initializer=Constant(value=0.1),
                #name='own_first_FC')(x)
	#x = Dense(4096, activation='relu')(x)

else:	# complete VGG is used
	# Simply get last FC output and change the classifier (done once outside 'if-else')
	x = base_model.layers[-2].output	
	
x = Dense(8, activation='softmax',name='predictions')(x)

model = Model(inputs=base_model.input, outputs=x)

print("")
print("Complete model (VGG16 + modifications) summary:")
print("")
model.summary()
plot_model(model, to_file='modelVGG16b_up2block=' + str(blocks_used) + '.png', show_shapes=True, show_layer_names=True)

if not train_from_scratch:
	for layer in base_model.layers:
		layer.trainable = False


model.compile(loss='categorical_crossentropy',optimizer='adadelta', metrics=['accuracy'])
for layer in model.layers:
	print(layer.name, layer.trainable)

# Define checkpoint to store the best model only
model_checkpoint = ModelCheckpoint(model_fname, monitor='val_acc', verbose=1,
			save_best_only=True, mode='max')
# Include TensorBoard here!
# When the modelling has started training, open a new terminal and enter:
# tensorboard --logdir ./TensorBoard_graphs (open the URL provided)
tbCallback = TensorBoard(log_dir='./TensorBoard_graphs', histogram_freq=0,
				write_graph=True, write_images=True)

# Early stopping, to avoid having to select the nb_epochs as a parameter,
# configure early stopping and stop after X epochs of worse val_acc
earlyStopping = EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='max')
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
# Change of params of 'fit_generator':
# DEPRECATED IN 2.1.2(nº of batches(steps)per epoch instead) ==> samples_per_epoch=batch_size*(int(400*1881/1881//batch_size)+1),
# Just feed the number of batches as int(n_examples//batch_size +1) where n_examples are the unique samples of training
history=model.fit_generator(train_generator,
		steps_per_epoch=int(n_examples//batch_size +1),
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
