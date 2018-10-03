# -*- coding: utf-8 -*-

import math
from random import shuffle
import os

import cv2
import pandas as pd
import numpy as np
from tensorflow.keras.utils import Sequence, to_categorical

class DataGenerator(Sequence):
	IMAGE_SIZE = 256
	IMAGE_CHANNELS = 3
	NUM_CLASSES = 2

	def __init__(self, train_data_dir, batch_size=2, num_samples=5, shuffle=True, useInvalid=False):
		""" initialize generator
		Args:
			train_data_dir:	path of directory that contains image data and labels
			batch_size:		number of samples per batch
			num_samples:	number of sample that the generator can use to form batches
			shuffle:		shuffle indexes after each epoch
			useInvalid:		use segments even if the bounding box is outside of the image
		"""
		self.train_data_dir = train_data_dir
		self.batch_size = batch_size
		self.num_samples = num_samples
		self.shuffle = shuffle

		self.dataframe = pd.read_csv(os.path.join(train_data_dir, 'labels.csv'), sep=';')

		sample_count = self.dataframe.shape[0]
		self.indexes = np.arange(sample_count)

		# allocate memory for input and output data array
		self.data_x = np.zeros([num_samples, self.IMAGE_SIZE, self.IMAGE_SIZE, self.IMAGE_CHANNELS])
		self.data_y = np.zeros([num_samples, 4], dtype='uint16')
		# generate data
		i = 0
		while i < num_samples:
			# cycle through available samples
			index = i % sample_count
			# read image
			path = os.path.join(self.train_data_dir, self.dataframe['path'].ix[index]) + '.jpg'
			
			image = cv2.imread(path)
			# get box coordinates
			box = self.parse_box(self.dataframe['price_box'].ix[index])

			valid, segment_image, segment_box = self.crop(image, box)

			if valid or useInvalid:
				# store data if valid
				self.data_x[i] = segment_image
				# to save memory, only store box coordinates
				# the full label will be generated later
				self.data_y[i] = segment_box

				# increment counter
				i += 1


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
		"""
		# get shuffled indexes for batch
		indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
		
		# input data
		x = np.array([ self.data_x[i] for i in indexes ])

		y = []
		for i in indexes:
			mask = np.zeros([self.IMAGE_SIZE, self.IMAGE_SIZE, self.NUM_CLASSES])
			
			# create class matrix, a class is assigned to each pixel
			temp = np.zeros([self.IMAGE_SIZE, self.IMAGE_SIZE], dtype='uint8')	
			bx, by, bw, bh = self.data_y[i]
			# assign class 2 to each pixel that is inside of the bounding box
			temp[by:by+bh, bx:bx+bw] = 1

			# transform into binarized category matrix
			y.append(to_categorical(temp, num_classes=self.NUM_CLASSES))

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

		scale_factor = self.IMAGE_SIZE / float(new_size)
		x = int((box[0] - offset_x) * scale_factor)
		y = int((box[1] - offset_y) * scale_factor)
		w = int(box[2] * scale_factor)
		h = int(box[3] * scale_factor)

		segment_img = cv2.resize(segment_img, (self.IMAGE_SIZE, self.IMAGE_SIZE))
		segment_box = [x, y, w, h]

		# segment is valid if the bounding box is not outside of the segment
		valid = not(x < 0 or y < 0 or x + w > self.IMAGE_SIZE or y + h > self.IMAGE_SIZE)
		
		return valid, segment_img, segment_box


	def on_epoch_end(self):
		""" shuffle indexes after each epoch """
		if self.shuffle == True:
			np.random.shuffle(self.indexes)


