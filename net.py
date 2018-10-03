#import warnings
#warnings.filterwarnings("ignore", message="numpy.dtype size changed")
#warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import numpy as np
import os


from utils import find_checkpoint, TensorboardWrapper

# keras imports
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Dense, Dropout, Flatten, Reshape
from tensorflow.keras.layers import Conv2D as Conv
from tensorflow.keras.layers import Conv2DTranspose as Deconv
from
#from keras.layers import MaxPooling1D as MaxPool
#from keras.layers import BatchNormalization as BatchNorm
#from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
#from keras.constraints import max_norm


class Model(object):
	
	def __init__(self, params, args):
		self.params = params
		self.args = args
		# build and compile model
		self.model = self.build()

	def build(self):
		model = Sequential()
		#256
		model.add(Conv(25, (3, 3), padding='same', activation='relu', input_shape=[256, 256, 3]))
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
		model.add(Dense(1024))
		model.add(Reshape(4, 4, 100))
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

		optimizer = RMSprop(lr=LEARNING_RATE, decay=LEARNING_RATE_DECAY)

		model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

		return model

	def load_checkpoint():
		if args.use_checkpoint:
			load_checkpoint(CHECKPOINT_DIR, args.model_name)

	def train(self, generator):
		#checkpoint_callback = ModelCheckpoint(os.path.join(CHECKPOINT_DIR, 'model.%s.{epoch:02d}.hdf5' % args.model_name), period=50)
		#tensorboard_callback = TensorBoardWrapper(validation_generator, log_dir=os.path.join(LOG_DIR, 'run_' + args.model_name), batch_size=BATCH_SIZE, histogram_freq=10)
		self.model.fit_generator(generator, epochs=500, verbose=1)

# database constants
CHECKPOINT_DIR = 'checkpoints'
LOG_DIR = 'logs'
EPOCHS = 500 
BATCH_SIZE = 12
MAX_NORM_LIMIT = 2.0
LEARNING_RATE = 0.001
LEARNING_RATE_DECAY = 0.05
MOMENTUM = 0.9

