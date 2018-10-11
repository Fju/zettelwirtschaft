#import warnings
#warnings.filterwarnings("ignore", message="numpy.dtype size changed")
#warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import numpy as np
import cv2
import os


from utils import find_checkpoint, TensorBoardWrapper

import tensorflow as tf
# keras imports
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Dense, Dropout, Flatten, Reshape
from tensorflow.keras.layers import Conv2D as Conv
from tensorflow.keras.layers import Conv2DTranspose as Deconv
#from keras.layers import MaxPooling1D as MaxPool
#from keras.layers import BatchNormalization as BatchNorm
from tensorflow.keras.callbacks import ModelCheckpoint

import time


class Model(object):
	
	def __init__(self, model_name, params):
		self.model_name = model_name
		self.params = params
		# build and compile model
		self.model = self.build()

	def build(self):
		image_size = self.params['image_size']
		
		model = Sequential()
		#256
		model.add(Conv(25, (3, 3), padding='same', activation='relu', input_shape=[image_size, image_size, 3]))
		model.add(Conv(50, (3, 3), strides=(2, 2), padding='same', activation='relu'))
		#128
		model.add(Conv(50, (3, 3), strides=(2, 2), padding='same', activation='relu'))
		#64
		model.add(Conv(50, (3, 3), strides=(2, 2), padding='same', activation='relu'))
		#32
		model.add(Conv(50, (3, 3), strides=(2, 2), padding='same', activation='relu'))
		#16
		model.add(Conv(100, (3, 3), strides=(2, 2), padding='same', activation='relu'))
		#8
		model.add(Conv(100, (3, 3), strides=(2, 2), padding='same', activation='relu'))
		#4
		model.add(Flatten())
		model.add(Dense(1600))
		model.add(Reshape([4, 4, 100]))
		model.add(Dropout(0.5))
		#4
		model.add(Deconv(100, (3, 3), strides=(2, 2), padding='same', activation='relu'))
		#8
		model.add(Deconv(100, (3, 3), strides=(2, 2), padding='same', activation='relu'))
		#16
		model.add(Deconv(50, (3, 3), strides=(2, 2), padding='same', activation='relu'))
		#32
		model.add(Deconv(50, (3, 3), strides=(2, 2), padding='same', activation='relu'))
		#64
		model.add(Deconv(50, (3, 3), strides=(2, 2), padding='same', activation='relu'))
		#128
		model.add(Deconv(50, (3, 3), strides=(2, 2), padding='same', activation='relu'))
		#256
		model.add(Deconv(2, (3, 3), padding='same', activation='softmax'))

		optimizer = RMSprop(lr=self.params['learning_rate'], decay=self.params['learning_rate_decay'])

		#model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

		return model

	def load_checkpoint(self):
		cp = find_checkpoint(self.params['checkpoint_dir'], self.model_name)
		self.model.load_weights(cp)

	def train(self, generator):
		checkpoint_callback = ModelCheckpoint(
			os.path.join(self.params['checkpoint_dir'], 'model.%s.{epoch:02d}.h5' % self.model_name),
			period=50
		)

		tensorboard_callback = TensorBoardWrapper(
			generator,
			log_dir=os.path.join(self.params['log_dir'], 'run_' + self.model_name),
			batch_size=self.params['batch_size']
		#	histogram_freq=10
		)

		self.model.fit_generator(
			generator,
			epochs=self.params['epochs'],
			verbose=self.params['verbosity'],
			class_weight='auto',
			callbacks=[checkpoint_callback, tensorboard_callback]
		)

	def evaluate(self, image):
		image_size = self.params['image_size']

		image = cv2.resize(image, (image_size, image_size))
		orig_image = np.array(image, copy=True)
 
		image = np.reshape(image, [1, image_size, image_size, 3]) / 128.0 - 1.0


		start = time.time()
		predictions = self.model.predict(image, batch_size=1)
		duration = time.time() - start

		print('computation time: %.1f ms' % (duration * 1000))


		predictions = np.resize(predictions, [image_size, image_size, 2])

		mask = np.zeros([image_size, image_size, 3])
		for y in range(image_size):
			for x in range(image_size):
				a, b = predictions[y, x, :]
				if a < b:
					mask[y, x, :] = 255.0

		cv2.imshow('orig', orig_image)
		cv2.imshow('mask', mask)
		cv2.waitKey()


