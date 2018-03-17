# -*- coding: utf-8 -*-

import cv2
import numpy as np
import math
from random import random as rand
from random import shuffle

class DataEntry(object):
	def __init__(self, image):
		self.image = image
		self.boxes = []

	def addBox(self, coordinates, class_num):
		""" add bounding box to list
		Args:
			coordinates: 1-D Array with four elements (x, y, w, h)
			class_num: Integer describing the class of the bounding box
		"""
		# turn coordinates of corners into width, height, and coordinates of center point
		w = coordinates[2]
		h = coordinates[3]
		cx = coordinates[0] + w / 2
		cy = coordinates[1] + h / 2
		# append box
		self.boxes.append([cx, cy, w, h, class_num])

	def _crop(self, image, size, pos):
		""" helper function to crop a numpy image
		Args:
			image: numpy array with pixel data
			size: 1-D array with two Integers (width, height), size of cropped segment
			pos: 1-D array with two Integers (x, y), offset segment in the original image
		Returns:
			segment: cropped image segment
		"""
		w, h = size.astype(np.int32, copy=True)
		x, y = pos.astype(np.int32, copy=True)
		# crop
		return image[y:y+h,x:x+w,:]

	def _rotate(self, angle):
		""" helper function to rotate a numpy image and bounding boxes
		Args:
			angle: rotation angle in degrees
		Returns:
			rotated_image: 3-D numpy array of rotated image
			rotated_boxes: 2-D array of bounding boxes
		"""
		rotated_boxes = []

		height, width = self.image.shape[:2]
		# center point is transform origin of rotation
		center = (width / 2, height / 2)

		for box in self.boxes:
			# receive bounding box coordinates
			cx, cy, w, h = box[:4]
			
			left, top, right, bottom = [cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5]
			edges = np.array([[left, top], [left, bottom], [right, top], [right, bottom]])
			# start values are set to always be bigger/smaller than an occuring minimal/maximal x/y value
			min_x, min_y = [width, height]
			max_x, max_y = [0, 0]
			for edge in edges:
				# deltas			
				dx, dy = edge - center
				# euclidean distance
				distance = math.sqrt(dx * dx + dy * dy)
				# calculate current angle
				alpha = math.atan2(dy, dx)
				# compute new position after rotation
				x = round(center[0] + math.cos(alpha - math.radians(angle)) * distance)
				y = round(center[1] + math.sin(alpha - math.radians(angle)) * distance)
				# store extremal values
				min_x = min(min_x, x)
				min_y = min(min_y, y)
				max_x = max(max_x, x)
				max_y = max(max_y, y)
			
			# modifiy bounding box coordinates
			w = max_x - min_x
			h = max_y - min_y
			cx = min_x + w / 2
			cy = min_y + h / 2
			# append bounding box to list
			rotated_boxes.append([cx, cy, w, h, box[4]])
		
		# create 2D rotation matrix rotating around center point by `angle` degress (0...360deg)
		matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
		rotated_image = cv2.warpAffine(self.image, matrix, (width, height))

		return rotated_image, rotated_boxes

	def rand_crop_factor(self):
		""" create a random number with modified distribution
		Returns:
			value: random float between 0 and 1
		"""
		
		return math.sqrt(math.sqrt(rand()))

	def rand_darken_factor(self):
		""" create a random number with modified distribution
		Returns:
			value: random float between 0.6 and 0.9
		"""
		return rand() * 0.3 + 0.6

	def augment(self, image_size, rotate=0.0, blur=0.0, darken=0.0, keepEmpty=False):
		""" generate a square segment and manipulate image randomly
		Args:
			image_size: Integer of pixels
			rotate: probabilty that the image will be rotated (keep 0.0 to prevent rotation)
			blur: probability that the image will be blurred (keep 0.0 to prevent blurring)
			keepEmpty: Boolean, if False a segment without boxes will be ignored
		Returns:
			valid: Boolean, if True the returned segment is valid otherwise the segment contains no information
			segment: 3-D array, whose width and height equal to `image_size`
			segment_boxes: 1-D array containing center point location, box coordinates and class (cx, cy, w, h, class_num)
			box_count: amount of bounding boxes that are inside the image segment
		"""
			
		image = self.image
		boxes = self.boxes
	
		if rand() < rotate:
			# use function y = 5(2x-1)², modified distribution of random values, x-range=[0...1], y-range=[-5...5]
			x = 2 * rand() - 1
			angle = 5 * x * x
			if angle != 0:
				# receive rotated image and rotated bounding boxes from method
				image, boxes = self._rotate(angle)

		if rand() < blur:
			# use function y = 5x² for modified distribution of random values, x-range=[0...1], y-range=[0...5]
			# values of 0 (no blur) are more common (44%) than higher values (10% ~ 20%)
			x = rand()
			y = int(5 * x * x)
			# don't apply blur if y = 0
			if y != 0:
				# blur image, averaging with quadratic kernel of size `y`
				image = cv2.blur(image, (y, y))

		if rand() < darken:
			image = image * self.rand_darken_factor()

		# get image dimension, reverse array so that (height, width) becomes (width, height)
		dim = np.array(image.shape[:2])[::-1] * 1.0
		# find smallest dimension and turn it into float
		min_dim = np.min(dim) * 1.0
		# smallest factor we can multiply the original image size with to not be smaller than `image_size`
		min_crop_factor = image_size / min_dim
		# generate a random factor for resizing the original image, factor is a float number between `min_crop_factor` and 1
		crop_factor = min_crop_factor + (1 - min_crop_factor) * self.rand_crop_factor()

		# dimensions of segment
		segment_size = np.floor(np.multiply([min_dim, min_dim], crop_factor))
		# random position of segment
		segment_pos = np.floor(np.multiply(dim - segment_size, [rand(), rand()]))
		
		# crop segment, use internal help function `_crop`	
		segment = self._crop(image, segment_size, segment_pos)
		# resize cropped segment to desired image size
		segment = cv2.resize(segment, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
		
		# double array content, e. g.: (x, y) becomes (x, y, x, y)
		scale_factor = np.divide(image_size, segment_size)
	
		
		segment_boxes = []
		for box in boxes:
			# adapt bounding box coordinates to resized and cropped segment
			# apply `segment_pos` as an offset to the center point (first two items of box array)
			# and multiply scale factor to both center point location and image dimension (third and fourth item of box array)
			cx, cy = np.multiply(box[:2] - segment_pos, scale_factor)
			w, h = np.multiply(box[2:4], scale_factor)
			
			# check if bounding box is inside segment (width and height are
			if cx - w / 2.0 > 0 and cx + w / 2.0 < image_size and cy - h / 2.0 > 0 and cy + h / 2.0 < image_size:
				# box is inside new segment, class number stays the same - obviously
				segment_boxes.append([round(cx), round(cy), round(w), round(h), box[4]])
		
		box_count = len(segment_boxes)

		if not(keepEmpty) and box_count == 0:
			# return error because the segment doesn't contain any boxes that could be detected
			# this behaviour depends on the `keepEmpty` flag
			return [False] * 4
		else:
			# other wise we return image data, box data and box count
			return True, segment, segment_boxes, box_count



# constants
BOX_THICKNESS	= 2
BOX_COLOR		= (170, 200, 0)
WINDOW_NAME		= 'Dataset Investigator'

class DatasetBuilder(object):
	# class enumeration
	CLASS_LOGO	= 0
	CLASS_PRICE	= 1
	CLASS_DATE	= 2


	def __init__(self, label_path, data_dir, params):
		self.label_path = label_path
		self.data_dir = data_dir
		self.data = []
		self.batch_size = int(params['batch_size'])
		self.epochs = int(params['epochs'])
		self.max_objects = int(params['max_objects'])
		self.image_size = int(params['image_size'])
		self.cell_count = int(params['cell_count'])

		self.batch_pos = 0
		self.epochs_pos = 0

		self.latest_batch = []
		# load data
		self._load()

	def _load(self):
		pos = 0
		label_file = open(self.data_dir + '/' + self.label_path, 'r')
		for line in label_file:
			# ignore comments
			if line.startswith('#'):
				continue
	
			# split line by ';', ignoring \n at the end		
			attr = line[:-1].split(';')
			for i in range(len(attr)):
				a = attr[i].split(',')
				if len(a) != 1:
					b = []
					for d in a:
						b.append(int(d))
					attr[i] = b

			if pos != 0:
				# skip header for now
				image = cv2.imread('%s/%s.jpg' % (self.data_dir, attr[0]))
				image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
				entry = DataEntry(image)
				for i in range(1, 4):
					entry.addBox(attr[i], i - 1)
				
				self.data.append(entry)

			# increment line position
			pos += 1


	def investigate(self, images, labels, box_cnt):
		""" creating a window showing the single images and respective bounding boxes of the current datasets images
		Args:
			images:		4-D ndarray [batch_size, image_size, image_size, 3]
			labels:		3-D ndarray [batch_size, max_objects, 5] (cx, cy, width, height, class_num)
			box_cnt:	1-D ndarray [batch_size]
		"""
		
		pos = 0
		while True:
			canvas = images[pos] # load current image, use it as canvas to draw bounding boxes onto it

			for i in range(box_cnt[pos]):
				cx, cy, w, h = labels[pos, i, :4]
				# draw bounding box
				cv2.rectangle(canvas, (int(cx - w / 2), int(cy - h / 2)), (int(cx + w / 2), int(cy + h / 2)), BOX_COLOR, BOX_THICKNESS)
	
			# show image	
			cv2.imshow(WINDOW_NAME, canvas)

			key = cv2.waitKey(0)
			if key == 1113939:
				# right arrow key, move to next image
				pos += 1
			elif key == 1113937:
				# left arrow key, move to previous image
				pos -= 1
			elif key == 1048603:
				# escape key, quit loop
				break

			# prevent out of bounds
			pos = (pos + self.batch_size) % self.batch_size
		
		# quit investigator
		cv2.destroyAllWindows()

	def validate(self, image, predicts, threshold=0.01):
		""" creating a window showing the predicted results on a test image, can be used for validation of the network
		Args:
			image:		image that was processed by the network
			predicts:	the networks predictions
			threshold:	confidence must be higher than this in order to display the prediction
		"""
		class_probs = predicts[0, :, :, :3]
		confidence = predicts[0, :, :, 3:4]
		coordinates = predicts[0, :, :, 4:]

		class_probs = np.reshape(class_probs, (7, 7, 3))
		confidence = np.reshape(confidence, (7, 7, 1))

		P = confidence * class_probs

		coordinates = np.reshape(coordinates, (7, 7, 4))

		# find boxes with enough confidence
		indexes = []
		for y in range(7):
			for x in range(7):
				class_index = np.unravel_index(np.argmax(P[y,x]), (3))
				
				confidence = P[y,x,class_index]
				print(confidence)
				if confidence > threshold:
					print('found likely box %.3f' % confidence)
					indexes.append([class_index, x, y])
		
	
		for class_num, x, y in indexes:
			cx, cy, w, h = coordinates[y,x]
			
			cx = (x + cx) * (self.image_size / self.cell_count)
			cy = (y + cy) * (self.image_size / self.cell_count)

			w = w * self.image_size
			h = h * self.image_size

			left = int(cx - w / 2.0)
			top = int(cy - h / 2.0)
			right = int(left + w)
			bottom = int(top + h)

			cv2.rectangle(image, left, top, right, bottom, (0, 0, 255))
			cv2.putText(image, str(class_num), left, top, 2, 1.5, (0, 0, 255))

		cv2.imshow('Validation', image)
		cv2.waitKey(0)


	def batch(self, investigate=False):
		""" generate batch containing images and related labels of `image_size`
			returns training data
		Args:
			investigate: boolean, if True it lets the user investigate the dataset with a minimal GUI
		Returns:
			images: 4-D ndarray [batch_size, image_size, image_size, 3]
			labels: 3-D ndarray [batch_size, max_objects, 5]
			objects_num: 1-D ndarray [batch_size]
		"""
	
		self.epochs_pos = (self.epochs_pos + 1) % self.epochs

		if self.epochs_pos != 1:
			# return previous batch if we haven't iterated `epochs` times
			return self.latest_batch


		# otherwise generate new batch		
		pos = self.batch_pos
		raw_batch = []
		while pos != self.batch_size:
			# pick one entry
			entry = self.data[pos % len(self.data)]
			# augment
			valid, image, boxes, cnt = entry.augment(self.image_size, rotate=0.25, blur=0.66, darken=0.66, keepEmpty=False)
			if valid:		
				raw_batch.append([image, boxes, cnt])
				pos += 1
		
		self.batch_pos = pos % len(self.data)
	
		# allocate memory
		b_images = np.zeros([self.batch_size, self.image_size, self.image_size, 3], dtype=np.float32)
		b_labels = np.zeros([self.batch_size, self.max_objects, 5], dtype=np.float32)
		b_box_count = np.zeros([self.batch_size], dtype=np.float32)

		shuffle(raw_batch)
		
		for i in range(len(raw_batch)):
			image, boxes, cnt = raw_batch[i]

			labels = np.zeros([self.max_objects, 5], dtype=np.float32)
			for j in range(cnt):
				labels[j] = boxes[j]
			
			image = image.astype(np.float32)
			image = 1.0 - image / 255.0

			b_labels[i,:,:] = labels
			b_images[i,:,:,:] = image
			b_box_count[i] = cnt

		# store as latest_batch
		self.latest_batch = [b_images, b_labels, b_box_count]

		if investigate:
			# let user investigate newly created batch
			self.investigate(b_images, b_labels, b_box_count)

		return b_images, b_labels, b_box_count

