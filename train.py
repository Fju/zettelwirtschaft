import tensorflow as tf
import numpy as np
import cv2
import time
import sys
import os

#from tensorflow.python import debug as tf_debug


class Solver(object):
 
	def __init__(self, net, dataset_builder, params):
		#process params
		self.learning_rate_decay = float(params['learning_rate_decay'])
		self.learning_rate = float(params['learning_rate'])
		self.batch_size = int(params['batch_size'])
		self.max_objects = int(params['max_objects'])
		self.image_size = int(params['image_size'])
		self.checkpoint_dir = 'checkpoints/'
		self.summary_dir = 'summary/'
		self.max_iterations = int(params['max_iterations'])
		self.epochs = int(params['epochs'])
		self.investigate = params['investigate']

		self.net = net
		self.dataset_builder = dataset_builder

	def _train(self):
		""" create an optimizer and apply to all trainable variables.
		Returns:
			train_op: operation for backpropagation
		"""

		lr = tf.train.exponential_decay(self.learning_rate, self.global_step, self.epochs, self.learning_rate_decay, staircase=True)
		opt = tf.train.AdagradOptimizer(lr)
		grads = opt.compute_gradients(self.total_loss)

		tf.summary.scalar('learning_rate', lr)

		apply_gradient_op = opt.apply_gradients(grads, global_step=self.global_step)

		return apply_gradient_op

	def construct_graph(self):
		""" defining input placeholders, build network and link with loss and train operation """
		self.global_step = self.net.create_global_step()

		self.images = tf.placeholder(tf.float32, (self.batch_size, self.image_size, self.image_size, 3))
		self.labels = tf.placeholder(tf.float32, (self.batch_size, self.max_objects, 5))
		self.object_counts = tf.placeholder(tf.int32, (self.batch_size))

		self.predictions = self.net.build(self.images)
		self.total_loss = self.net.loss(self.predictions, self.labels, self.object_counts)
		
		tf.summary.scalar('loss', self.total_loss)
		self.train_op = self._train()


	def solve(self, restore):
		""" training operation, iterates through each epoch and saves checkpoints on it's way
		Args:
			restore: if `True` it restores the latest checkpoint, otherwise it starts training a "blank" network
		"""

		self.construct_graph()

		saver = tf.train.Saver(self.net.var_collection, write_version=tf.train.SaverDef.V2)
		
		# create session variable
		sess = tf.Session()
		# init all variables
		sess.run(tf.global_variables_initializer())

		# summary_writer is used to log the important loss values or the learning rate for easy insights
		summary_writer = tf.summary.FileWriter(self.summary_dir, sess.graph)
		summary_op = tf.summary.merge_all()

		# find latest checkpoint
		latest_checkpoint = tf.train.latest_checkpoint(self.checkpoint_dir)
		# restor if there is a checkpoint and the user wants it to
		if latest_checkpoint != None and restore:
			print('Restoring model (path: "%s")' % latest_checkpoint)
			saver.restore(sess, latest_checkpoint)
		else:
			print('No model restored')

		# use global_step of restored model, if no model was restored it returns 0 by default
		step = tf.train.global_step(sess, self.global_step)
		while step < self.max_iterations:
			start_time = time.time()
			b_images, b_labels, b_object_counts = self.dataset_builder.batch(self.investigate)
		
			_, summary_str, loss_value = sess.run([self.train_op, summary_op, self.total_loss], feed_dict={self.images: b_images, self.labels: b_labels, self.object_counts: b_object_counts})
			#loss_value, nilboy = sess.run([self.total_loss, self.nilboy], feed_dict={self.images: np_images, self.labels: np_labels, self.objects_num: np_objects_num})

			duration = time.time() - start_time

			assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
	
			step = tf.train.global_step(sess, self.global_step)

			num_examples_per_step = self.batch_size
			examples_per_sec = num_examples_per_step / duration
			sec_per_batch = float(duration)

			format_str = ('step %d, loss = %.4f (%.1f examples/sec; %.3f sec/batch)')
			print(format_str % (step, loss_value, examples_per_sec, sec_per_batch))

			if step % 10 == 0:
				summary_writer.add_summary(summary_str, step)
			
			if step % 200 == 0 and step != 0:
				print('Saving checkpoint')
				saver.save(sess, self.checkpoint_dir + 'model.ckpt', global_step=self.global_step)
		
		# save final checkpoint
		print('Finished training')
		saver.save(sess, self.checkpoint_dir + 'model.ckpt', global_step=self.global_step)

		sys.stdout.flush()
		sess.close()

	def validate(self, image_path):
		sess = tf.Session()
		
		image = tf.placeholder(tf.float32, (1, self.image_size, self.image_size, 3))
		predicts = self.net.build(image, dropout=0.0)

		sess.run(tf.global_variables_initializer())
	
		saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)

		latest_checkpoint = tf.train.latest_checkpoint('checkpoints/')
		if latest_checkpoint == None:
			print('No checkpoint found, cannot validate')
			return
	
		saver.restore(sess, latest_checkpoint)

		np_img = cv2.imread(image_path)
		np_img = cv2.resize(np_img, (self.image_size, self.image_size))
		np_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)

		inv_img = np_img.astype(np.float32)
		inv_img = 1.0 - inv_img / 255.0
		inv_img = np.reshape(inv_img, (1, self.image_size, self.image_size, 3))

		start = time.time()
		result = sess.run(predicts, feed_dict={image: inv_img})
		duration = time.time() - start
	
		print('Computation time: %.2fms' % (duration * 1000))
		
		return np_img, result

