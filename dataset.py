# -*- coding: utf-8 -*-

import math
from random import shuffle
import os

import cv2
import pandas as pd
import numpy as np
from tensorflow.keras.utils import Sequence, to_categorical
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

class DataGenerator(Sequence):

	def __init__(self, params, shuffle=True, useInvalid=False):
		""" initialize generator
		Args:
			params:			dictionary containing global params
			shuffle:		shuffle indexes after each epoch
			useInvalid:		use segments even if the bounding box is outside of the image
		"""
		self.batch_size = params['batch_size']
		self.num_samples = params['num_samples']
		self.image_size = params['image_size']
		self.shuffle = shuffle

		train_data_dir = params['train_data_dir']

		print('generating data...')
		dataframe = pd.read_csv(os.path.join(train_data_dir, 'labels.csv'), sep=';')
		sample_count = dataframe.shape[0]

		self.indexes = np.arange(self.num_samples)

		# allocate memory for input and output data array
		self.data_x = np.zeros([self.num_samples, self.image_size, self.image_size, 3])
		self.data_y = np.zeros([self.num_samples, 4], dtype='uint16')
		# generate data
		i = 0
		while i < self.num_samples:
			# cycle through available samples
			index = i % sample_count
			# read image
			path = os.path.join(train_data_dir, dataframe['path'].ix[index]) + '.jpg'
			
			image = cv2.imread(path)
			# get box coordinates
			box = self.parse_box(dataframe['price_box'].ix[index])

			valid, segment_image, segment_box = self.crop(image, box)

			if valid or useInvalid:
				# store data if valid
				self.data_x[i] = segment_image
				# to save memory, only store box coordinates
				# the full label will be generated later
				self.data_y[i] = segment_box

				# increment counter
				i += 1
		print('done')

	def __len__(self):
		""" calculates number of batches per epoch """
		return int(self.num_samples / self.batch_size)

	def __getitem__(self, index):
		""" generate one batch
		Args:
			index:	number between 0 and the value returned by the __len__ function
		Returns:
			x:		input data [batch_size, image_size, image_size, image_channels]
			y:		label data [batch_size, image_size, image_size, num_classes]
			sample_weights
		"""
		# get shuffled indexes for batch
		indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
		
		# input data
		x = np.array([ self.data_x[i] for i in indexes ])

		y = []
		for i in indexes:
			mask = np.zeros([self.image_size, self.image_size, 2])
			
			# create class matrix, a class is assigned to each pixel
			temp = np.zeros([self.image_size, self.image_size], dtype='uint8')	
			bx, by, bw, bh = self.data_y[i]
			# assign class 2 to each pixel that is inside of the bounding box
			temp[by:by+bh, bx:bx+bw] = 1

			# transform into binarized category matrix
			mask = to_categorical(temp, num_classes=mask.shape[-1])
			
			y.append(mask)

		# cast to numpy
		y = np.array(y)
		

		return x, y

	def parse_box(self, value):
		""" parses string of coordinates (x, y, width, height) to array of integers
		Args:
			value:	string containing coordinates separated with commas
		Returns:
			box:	array of four integers
		"""
		box = [ int(b) for b in value.split(',') ]
		
		return box

	def crop(self, image, box, min_size=180):
		""" crops and scales raw image to generate an image segment and corresponding labeling
		Args:
			image:			raw image
			box:			raw box coordinates (x, y, width, height)
			min_size:		smallest possible window that can be used as a segment
		Returns:
			valid:			indicates whether the segment fully contains the bounding box
			segment_img:	cropped and scaled image matrix
			segment_box:	repositioned and scaled box coordinates (x, y, width, height)
		"""

		original_size = np.min(image.shape[:2])
		# new segment size can lie between minimum segment size and the original size
		new_size = np.random.randint(min_size, original_size)
		offset_x, offset_y = np.random.randint(0, original_size - new_size, size=2)

		segment_img = image[offset_y:offset_y+new_size, offset_x:offset_x+new_size, :]

		scale_factor = self.image_size / float(new_size)
		x = int((box[0] - offset_x) * scale_factor)
		y = int((box[1] - offset_y) * scale_factor)
		w = int(box[2] * scale_factor)
		h = int(box[3] * scale_factor)

		segment_img = cv2.resize(segment_img, (self.image_size, self.image_size))
		segment_box = [x, y, w, h]

		# segment is valid if the bounding box is not outside of the segment
		valid = not(x < 0 or y < 0 or x + w > self.image_size or y + h > self.image_size)
		
		return valid, segment_img, segment_box


	def on_epoch_end(self):
		""" shuffle indexes after each epoch """
		if self.shuffle == True:
			np.random.shuffle(self.indexes)

