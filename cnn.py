import tensorflow as tf
import numpy as np

class Net(object):
	def __init__(self, params):
		self.image_size = int(params['image_size'])
		self.num_classes = int(params['num_classes'])
		self.cell_count = int(params['cell_count'])
		self.boxes_per_cell = int(params['boxes_per_cell'])
		self.batch_size = int(params['batch_size'])

	
		self.class_scale = 1.0
		self.object_scale = 2.0
		self.noobject_scale = 1.0
		self.coord_scale = 5.0

		self.trainable_collection = []

		self.conv_layer_index = 0
		self.dense_layer_index = 0

	def _variable_on_cpu(self, name, shape, initializer):
		""" helper function to create a variable stored on CPU memory.
		Args:
			name: name of the Variable
			shape: list of ints
			initializer: initializer of Variable
		Returns:
			Variable Tensor
		"""
		with tf.device('/cpu:0'):
			var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)
			self.trainable_collection.append(var)
		return var 


	def _weight_var(self, shape, stddev=0.005, wd=0.0005):
		""" helper function to create a weight variable tensor, initialized with a random number
		Args:
			name:	name of the variable 
			shape: 	array of dimension sizes
			stddev:	standard deviation of a truncated Gaussian
			wd:		weight decay
		Returns:
			variable tensor 
		"""
		initializer = tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32)

		var = self._variable_on_cpu('weights', shape, initializer)

		if wd is not None:
			weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
			tf.add_to_collection('losses', weight_decay)

		return var
	
	def _bias_var(self, shape, constant=0.01):
		""" helper function to create a bias variable, use low initial value to avoid dead neurons
		Args:
			name: name of variable
			shape: array of dimension sizes
			constant: balue of variable when initialized
		Returns:
			variable tensor
		"""
		initializer = tf.constant_initializer(constant, dtype=tf.float32)
		return self._variable_on_cpu('biases', shape, initializer)

	def global_step(self):
		with tf.variable_scope('global_step', reuse=tf.AUTO_REUSE):
			var = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
			return var	

	def conv2d(self, x, kernel, stride=1):
		""" create one convolution layer and init weight and bias variables
		Args:
			x: input tensor
			kernel: array with (filter_size, filter_size, input_depth, output_depth)
			stride: filter stride
		Returns:
			ready-to-use convolutional layer
		"""
		# increment layer index
		self.conv_layer_index += 1		
		with tf.variable_scope('conv%d' % self.conv_layer_index, reuse=tf.AUTO_REUSE) as scope:
			# init variable in specific scope
			weights = self._weight_var(kernel)
			biases = self._bias_var([kernel[-1]])
			# pipe input through convolution, addition of biases and ReLU
			return tf.nn.leaky_relu(tf.nn.bias_add(tf.nn.conv2d(x, weights, strides=[1, stride, stride, 1], padding='SAME'), biases), alpha=0.1)

	def max_pool(self, x, stride=2):
		""" create max pooling layer
		Args:
			x: input tensor
			stride: filter stride and filter size (stride = 2 reduces tensor dimension by half in second and third dimension)
		Returns:
			ready-to-use max pooling layer
		"""
		return tf.nn.max_pool(x, ksize=[1, stride, stride, 1], strides=[1, stride, stride, 1], padding='SAME')
	
	def dense(self, x, input_depth, output_depth):
		""" create fully connected / dense layer
		Args:
			x: input tensor
			input_depth: depth of input tensor
			output_depth: depth of output tensor
		Returns:
			ready-to-use dense layer
		"""
		# increment layer index
		self.dense_layer_index += 1
		with tf.variable_scope('dense%d' % self.dense_layer_index, reuse=tf.AUTO_REUSE) as scope:
			# init variable in specific scope
			weights = self._weight_var([input_depth, output_depth])
			biases = self._bias_var([output_depth])
			# pipe input through weight multiplication, addition of biases and ReLU
			return tf.nn.leaky_relu(tf.nn.bias_add(tf.matmul(x, weights), biases), alpha=0.1)

	def iou(self, p_boxes, t_box):
		""" calculate ious
		Args:
			p_boxes: predicted boxes 4-D tensor [cell_count, cell_count, boxes_per_cell, 4]	====> (x_center, y_center, w, h)
				t_box: true box 1-D tensor [4] ===> (x_center, y_center, w, h)
		Return:
			IoU: 3-D tensor [cell_count, cell_count, )boxes_per_cell]
		"""
		
		# convert boxes from [center_x, center_y, width, height] to [left, top, right, bottom]
		p_boxes = tf.stack([p_boxes[:, :, :, 0] - p_boxes[:, :, :, 2] / 2, p_boxes[:, :, :, 1] - p_boxes[:, :, :, 3] / 2,
							p_boxes[:, :, :, 0] + p_boxes[:, :, :, 2] / 2, p_boxes[:, :, :, 1] + p_boxes[:, :, :, 3] / 2])
		
		# transpose boxes tensor, so that 0th dimension (box coordinates) are the 3rd dimension
		p_boxes = tf.transpose(p_boxes, [1, 2, 3, 0])
		
		# convert true box from [center_x, center_y, width, height] to [left, top, right, bottom] 
		t_box = tf.stack([t_box[0] - t_box[2] / 2, t_box[1] - t_box[3] / 2,
							t_box[0] + t_box[2] / 2, t_box[1] + t_box[3] / 2])

		# find top-left point and bottom-right point (use max/min of true box or predicted box)
		top_left = tf.maximum(p_boxes[:, :, :, 0:2], t_box[0:2])	# [:, :, :, 2] ==> (left, top)
		bottom_right = tf.minimum(p_boxes[:, :, :, 2:], t_box[2:])	# [:, :, :, 2] ==> (right, bottom)

		# intersection [:, :, :, 2] ==> (right - left, bottom - top)
		intersection = bottom_right - top_left 

		# calculate area of intersection rectangle (A = a * b)
		inter_square = intersection[:, :, :, 0] * intersection[:, :, :, 1]

		# prevent negative area
		mask = tf.cast(intersection[:, :, :, 0] > 0, tf.float32) * tf.cast(intersection[:, :, :, 1] > 0, tf.float32)
		inter_square = mask * inter_square
		
		# calculate areas of rectangles (A = a * b)
		square1 = (p_boxes[:, :, :, 2] - p_boxes[:, :, :, 0]) * (p_boxes[:, :, :, 3] - p_boxes[:, :, :, 1])
		square2 = (t_box[2] - t_box[0]) * (t_box[3] - t_box[1])
		
		# IoU: Intersection over Union (see https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/)
		# Union is equal to the sum of both areas minus the intersection
		return inter_square / (square1 + square2 - inter_square + 1e-6)


	def build(self, images):
		""" build convolutional neural network
		Args:
			images: input images 4-D [batch_size, image_size, image_size, 3] ==> (r, g, b)
		Return:
			predicts: cnn results 4-D tensor
		"""

		self.dense_layer_index = 0
		self.conv_layer_index = 0
	
		# 448
		temp_conv = self.conv2d(images, [3, 3, 3, 64])
		temp_pool = self.max_pool(temp_conv)

		# 224
		temp_conv = self.conv2d(temp_pool, [3, 3, 64, 128])
		temp_pool = self.max_pool(temp_conv)
		
		# 112
		temp_conv = self.conv2d(temp_pool, [3, 3, 128, 128])
		temp_pool = self.max_pool(temp_conv)
		
		# 56
		temp_conv = self.conv2d(temp_pool, [3, 3, 128, 256])
		temp_pool = self.max_pool(temp_conv)

		# 28
		temp_conv = self.conv2d(temp_pool, [3, 3, 256, 256])
		temp_pool = self.max_pool(temp_conv)

		# 14
		temp_conv = self.conv2d(temp_pool, [3, 3, 256, 512])
		temp_pool = self.max_pool(temp_conv)

		# 7
		temp_conv = self.conv2d(temp_pool, [3, 3, 512, 512])
		temp_pool = self.conv2d(temp_conv, [3, 3, 512, 512])

		flat_layer = tf.reshape(temp_pool, [-1, self.cell_count * self.cell_count * 512])
		
		temp_dense = self.dense(flat_layer, self.cell_count * self.cell_count * 512, 4096)
		temp_dense = tf.nn.dropout(temp_dense, keep_prob=0.5)
		temp_dense = self.dense(temp_dense, 4096, self.cell_count * self.cell_count * (self.num_classes + self.boxes_per_cell * 5))
	
		n1 = self.cell_count * self.cell_count * self.num_classes # classification
		n2 = n1 + self.cell_count * self.cell_count * self.boxes_per_cell # box

		class_probs = tf.reshape(temp_dense[:, 0:n1], (-1, self.cell_count, self.cell_count, self.num_classes))
		confidence = tf.reshape(temp_dense[:, n1:n2], (-1, self.cell_count, self.cell_count, self.boxes_per_cell))
		boxes = tf.reshape(temp_dense[:, n2:], (-1, self.cell_count, self.cell_count, self.boxes_per_cell * 4))

		# apply sigmoid function to center coordinates and exponential function to width and height of the bounding box
		# described in the "YOLO9000:Better, Faster, Stronger" documentation (source: https://arxiv.org/pdf/1612.08242.pdf, page 4)
		centers = tf.nn.sigmoid(boxes[:,:,:,0:2])
		sizes = tf.exp(boxes[:,:,:,2:4])

		boxes = tf.Print(boxes, [tf.shape(centers), tf.shape(sizes)], 'Shape of splitted tensors', summarize=5)
	
		boxes = tf.concat([centers, sizes], 3)

		predictions = tf.concat([class_probs, confidence, boxes], 3)
		
		return predictions

	def loss_loop_cond(self, cnt, object_count, losses, prediction, labels):
		# keep looping when index is smaller than current object count
		return cnt < object_count

	def loss_loop_body(self, cnt, object_count, losses, prediction, labels):
		""" calculate loss of a single execution
		Args:
			predict: 3-D tensor [cell_count, cell_count, 5 * boxes_per_cell]
			labels : [max_objects, 5]	(x_center, y_center, w, h, class)
		"""
		# get current label, label contains center point coordinates (x, y)
		label = labels[cnt:cnt+1, :]
		label = tf.reshape(label, [-1])
	
		# store coordinates of true box		
		t_left = tf.floor((label[0] - label[2] / 2) / (self.image_size / self.cell_count))
		t_right = tf.ceil((label[0] + label[2] / 2) / (self.image_size / self.cell_count))
		t_top = tf.floor((label[1] - label[3] / 2) / (self.image_size / self.cell_count))
		t_bottom = tf.ceil((label[1] + label[3] / 2) / (self.image_size / self.cell_count))

		#t_bottom = tf.Print(t_bottom, [t_left, t_top, t_right, t_bottom], 'coordinates: ')
		# generate tensor of size [cell_count, cell_count] filled with zeros except the region where the object is in
		# 1. 1-D tensor containing width and height of true box (width = t_right - t_left, height = t_bottom - t_top)
		temp = tf.cast(tf.stack([t_bottom - t_top, t_right - t_left]), dtype=tf.int32)
		# 2. 2-D tensor of size [height, width] filled with 1.f
		t_region_mask = tf.ones(temp, tf.float32)
		# 3. 2-D tensor of size [2, 2] that contains paddings ((pad_top, pad_bottom), (pad_left, pad_right))
		temp = tf.reshape(tf.cast(tf.stack([t_top, self.cell_count - t_bottom, t_left, self.cell_count - t_right]), tf.int32), (2, 2))

		# 4. apply padding, resulting 2-D tensor has size [cell_count, cell_count], padded area is filled with zeros
		t_region_mask = tf.pad(t_region_mask, temp, "CONSTANT")

		# store center coordinates
		t_center_x, t_center_y = tf.floor(label[0] / (self.image_size / self.cell_count)), tf.floor(label[1] / (self.image_size / self.cell_count))

		# generate tensor of size [cell_count, cell_count] filled with zeros except the center of the true object
		# 1. 1-D tensor of size [1, 1]
		t_center_mask = tf.ones([1, 1], tf.float32)
		# 2. 2-D tensor of size [2, 2] that contains paddings ((pad_top, pad_bottom), (pad_left, pad_right))
		temp = tf.reshape(tf.cast(tf.stack([t_center_y, self.cell_count - t_center_y - 1, t_center_x, self.cell_count - t_center_x - 1]), tf.int32), (2, 2))
		# 3. apply padding, resulting 2-D tensor has size [cell_count, cell_count], padded area is filled with zeros
		t_center_mask = tf.pad(t_center_mask, temp, "CONSTANT")
	

		# store 3-D tensor of size [cell_count, cell_count, 4] ==> (center_x, center_y, width, height)
		predict_boxes = tf.reshape(prediction[:, :, self.num_classes + self.boxes_per_cell:], [self.cell_count, self.cell_count, self.boxes_per_cell, 4])

		# map relative predicted values to absolute image coordinates and sizes
		predict_boxes = predict_boxes * [self.image_size / self.cell_count, self.image_size / self.cell_count, self.image_size, self.image_size]
		
		
		# create 3-D numpy array for shifting relative center points by cell offset
		box_offsets = np.zeros([self.cell_count, self.cell_count, 4])
		for y in range(self.cell_count):
			for x in range(self.cell_count):
				# apply cell specific offset (no width or height offset)
				box_offsets[y, x, :] = [self.image_size / self.cell_count * x, self.image_size / self.cell_count * y, 0, 0]

		# repeat pattern by boxes_per_cell times along the 3rd dimension
		box_offsets = np.tile(np.resize(box_offsets, [self.cell_count, self.cell_count, 1, 4]), [1, 1, self.boxes_per_cell, 1])

		# apply bounding box offsets
		predict_boxes = box_offsets + predict_boxes

		# compute IoU scores as a 3-D tensor of size [cell_count, cell_count, boxes_per_cell]
		iou_predict_truth = self.iou(predict_boxes, label[0:4])

		# mask ious scores with center point mask, get class
		t_confidence = iou_predict_truth * tf.reshape(t_center_mask, [self.cell_count, self.cell_count, 1])

		# mask iou scores with center point mask
		masked_iou = iou_predict_truth * tf.reshape(t_center_mask, (self.cell_count, self.cell_count, 1))
		
		# last dimension is reduced to the highest element, output tensor has size [cell_count, cell_count, 1]
		max_iou = tf.reduce_max(masked_iou, 2, keepdims=True)

		masked_iou = tf.cast((masked_iou >= max_iou), tf.float32) * tf.reshape(t_center_mask, (self.cell_count, self.cell_count, 1))

		# invert masked_iou tensor (0 -> 1, 1 -> 0)
		masked_iou_inv = tf.ones_like(masked_iou, dtype=tf.float32) - masked_iou

		# get predicted confidence score, 3-D tensor of size [cell_count, cell_count, 1]
		p_confidence = prediction[:, :, self.num_classes:self.num_classes + self.boxes_per_cell]

		# store true bounding box coordinates in 0-D tensors
		t_x = label[0]
		t_y = label[1]
		t_sqrt_w = tf.sqrt(tf.abs(label[2]))
		t_sqrt_h = tf.sqrt(tf.abs(label[3]))

		# get predicted bounding box coordinates in 3-D tensors of size [cell_count, cell_count, boxes_per_cell]
		p_x = predict_boxes[:, :, :, 0]
		p_y = predict_boxes[:, :, :, 1]

		p_sqrt_w = tf.sqrt(tf.minimum(self.image_size * 1.0, tf.maximum(0.0, predict_boxes[:, :, :, 2])))
		p_sqrt_h = tf.sqrt(tf.minimum(self.image_size * 1.0, tf.maximum(0.0, predict_boxes[:, :, :, 3])))

		# create 1-D tensor of size [num_classes] filled with zeros except one 1 at the position of the true class
		t_class = tf.one_hot(tf.cast(label[4], tf.int32), self.num_classes, dtype=tf.float32)

		# get predicted class tensor of size [cell_count, cell_count, num_classes]
		p_class = prediction[:, :, 0:self.num_classes]

		# compute loss of predicting right class
		class_loss = tf.nn.l2_loss(tf.reshape(t_region_mask, (self.cell_count, self.cell_count, 1)) * (p_class - t_class)) * self.class_scale

		# compute loss of detecting objects correctly
		object_loss = tf.nn.l2_loss(masked_iou * (p_confidence - t_confidence)) * self.object_scale

		# compute loss of detecting empty cells with no objects
		noobject_loss = tf.nn.l2_loss(masked_iou_inv * (p_confidence)) * self.noobject_scale

		# compute loss of finding right coordinates
		coord_loss = (tf.nn.l2_loss(masked_iou * (p_x - t_x) / (self.image_size / self.cell_count)) +
					 tf.nn.l2_loss(masked_iou * (p_y - t_y) / (self.image_size / self.cell_count)) +
					 tf.nn.l2_loss(masked_iou * (p_sqrt_w - t_sqrt_w)) / self.image_size +
					 tf.nn.l2_loss(masked_iou * (p_sqrt_h - t_sqrt_h)) / self.image_size) * self.coord_scale

		return cnt + 1, object_count, [losses[0] + class_loss, losses[1] + object_loss, losses[2] + noobject_loss, losses[3] + coord_loss], prediction, labels

	def loss(self, predictions, labels, object_counts):
		""" Add Loss to all the trainable variables
		Args:
			predicts: 4-D tensor [batch_size, cell_count, cell_count, num_classes + 5 * boxes_per_cell]	==> (num_classes, boxes_per_cell, 4 * boxes_per_cell)
			labels: 3-D tensor of [batch_size, max_objects, 5]
			objects_count: 1-D tensor [batch_size]
		"""
		# loss variables
		class_loss = tf.constant(0, tf.float32)
		object_loss = tf.constant(0, tf.float32)
		noobject_loss = tf.constant(0, tf.float32)
		coord_loss = tf.constant(0, tf.float32)
		losses = [0, 0, 0, 0]

		# iterate through whole training batch
		for i in range(self.batch_size):
			# get current prediction tensor of size [cell_count, cell_count, num_classes + 5 * boxes_per_cell]
			current_prediction = predictions[i, :, :, :]
			# get current label tensor of size [max_objects, 5]
			current_label = labels[i, :, :]
			# 1-D tensor containing amount of true objects along the whole batch
			object_cnt = object_counts[i]
			# run detection for all objects the current item contains (`object_num` times)
			results = tf.while_loop(self.loss_loop_cond, self.loss_loop_body, [tf.constant(0), object_cnt, [class_loss, object_loss, noobject_loss, coord_loss], current_prediction, current_label])
			for j in range(4):
				# add computed losses to `losses` array
				losses[j] = losses[j] + results[2][j]

		losses_sum = losses[0] + losses[1] + losses[2] + losses[3]
	

		tf.add_to_collection('losses', losses_sum / self.batch_size)

		tf.summary.scalar('class_loss', losses[0] / self.batch_size)
		tf.summary.scalar('object_loss', losses[1] / self.batch_size)
		tf.summary.scalar('noobject_loss', losses[2] / self.batch_size)
		tf.summary.scalar('coord_loss', losses[3] / self.batch_size)
		tf.summary.scalar('weight_loss', tf.add_n(tf.get_collection('losses')) - losses_sum / self.batch_size)

		return tf.add_n(tf.get_collection('losses'), name='total_loss')

