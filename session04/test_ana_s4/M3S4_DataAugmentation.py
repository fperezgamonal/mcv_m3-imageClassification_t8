from keras.preprocessing.image import ImageDataGenerator
import os

class DataAugmentation:
	__slots__=  ['__samplewise_center', '__featurewise_std_normalization', '__samplewise_std_normalization', '__preprocess_input',
		'__rotation_range', '__width_shift_range','__height_shift_range','__shear_range','__zoom_range',
		'__channel_shift_range','__fill_mode', '__cval', '__horizontal_flip','__vertical_flip','__rescale',
		'__number_of_epoch','__batch_size','__img_width','__img_height',
		'__train_data_dir','__val_data_dir','__test_data_dir', '__model']
	def __init__ (self, preprocess_input, model):
		self.__samplewise_center = False
		self.__featurewise_std_normalization = False
		self.__samplewise_std_normalization = False
		self.__preprocess_input = preprocess_input
		self.__rotation_range = 0.
		self.__width_shift_range = 0.
		self.__height_shift_range = 0.
		self.__shear_range = 0.
		self.__zoom_range = 0.
		self.__channel_shift_range = 0.
		self.__fill_mode='nearest'
		self.__cval=0.
		self.__horizontal_flip = False 
		self.__vertical_flip = False
		self.__rescale =  None
		
		self.__number_of_epoch = 20
		self.__batch_size = 32 
		
		self.__img_width = 224
		self.__img_height = 224
		
		
		self.__train_data_dir = '../../Databases/MIT_split/train'
		self.__val_data_dir = '../../Databases/MIT_split/test'
		self.__test_data_dir = '../../Databases/MIT_split/test'

		self.__model = model
		
	def set_number_of_epoch (self,param):
		self.__number_of_epoch = param
	def set_batch_size(self,param):
		self.__batch_size = param		
		
	def set_samplewise_center (self,param):
		self.__samplewise_center = param
	def set_featurewise_std_normalization (self,param):
		self.__featurewise_std_normalization = param
	def set_samplewise_std_normalization (self,param):
		self.__samplewise_std_normalization = param
	def set_preprocess_input (self,param):
		self.__preprocess_input = param 
	def set_rotation_range (self,param):
		self.__rotation_range = param
	def set_width_shift_range (self,param):
		self.__width_shift_range = param
	def set_height_shift_range (self,param):
		self.__height_shift_range = param
	def set_shear_range (self,param):
		self.__shear_range = param
	def set_zoom_range (self,param):
		self.__zoom_range = param
	def set_channel_shift_range (self,param):
		self.__channel_shift_range = param
	def set_fill_mode (self,param):
		self.__fill_mode = param
	def set_cval (self,param):
		self.__cval = param
	def set_horizontal_flip (self,param):
		self.__horizontal_flip = param
	def set_vertical_flip (self,param):
		self.__vertical_flip = param
	def set_rescale (self, param):
		self.__rescale =  param
			
	def create(self):
		datagen = ImageDataGenerator(featurewise_center=False,
			samplewise_center = self.__samplewise_center,
			featurewise_std_normalization = self.__featurewise_std_normalization,
			samplewise_std_normalization = self.__samplewise_std_normalization,
			preprocessing_function = self.__preprocess_input,
			rotation_range = self.__rotation_range,
			width_shift_range = self.__width_shift_range,
			height_shift_range = self.__height_shift_range,
			shear_range = self.__shear_range,
			zoom_range = self.__zoom_range,
			channel_shift_range = self.__channel_shift_range,
			fill_mode=self.__fill_mode,
			cval=self.__cval,
			horizontal_flip=self.__horizontal_flip ,
			vertical_flip=self.__vertical_flip ,
			rescale=self.__rescale)		
			
		return datagen
	
	def flow (self, datagen, batch_size, img_width, img_height, val_data_dir, test_data_dir, train_data_dir):

		train_generator = datagen.flow_from_directory(train_data_dir,
			target_size=(img_width, img_height),
			batch_size=batch_size,
			class_mode='categorical',
			save_to_dir='images', 
			save_prefix='aug_train_', 
			save_format='png')

		test_generator = datagen.flow_from_directory(test_data_dir,
				target_size=(img_width, img_height),
				batch_size=batch_size,
				class_mode='categorical')

		validation_generator = datagen.flow_from_directory(val_data_dir,
				target_size=(img_width, img_height),
				batch_size=batch_size,
				class_mode='categorical')
				
		return train_generator, test_generator, validation_generator


	def fit (self, train_generator, validation_generator, number_of_epoch, batch_size):
	
		history=self.__model.fit_generator(train_generator,
			samples_per_epoch=batch_size*(int(400*1881/1881//batch_size)+1),
			nb_epoch=number_of_epoch,
			validation_data=validation_generator,
			nb_val_samples=807)
			
		return history
	
	def run (self):
		# create ImageDataGenerator
		print ('***')
		#if (os.path.exists('images')==False:
		os.makedirs('images')
		
		print ('step1')
		datagen = self.create()

		# configure the batch size and 
		# prepare the data generator and get batches of images
		print ('step2')
		train_generator, test_generator, validation_generator = self.flow(datagen, self.__batch_size, 
			self.__img_width, 
			self.__img_height, 
			self.__val_data_dir, 
			self.__test_data_dir, 
			self.__train_data_dir)
		
		# calculate any statistics required to actually 
		# perform the transforms to your image data
		print ('step3')
		history = self.fit(train_generator, 
			validation_generator, 
			self.__number_of_epoch, 
			self.__batch_size)
		print ('***')
		return history, test_generator 
		
	

