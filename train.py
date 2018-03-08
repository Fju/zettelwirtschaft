import tensorflow as tf
import numpy as np
import time
import sys
import os

from tensorflow.python import debug as tf_debug


class Solver(object):
 
	def __init__(self, net, dataset_builder, params):
		#process params
		self.moment = float(params['moment'])
		self.learning_rate = float(params['learning_rate'])
		self.batch_size = int(params['batch_size'])
		self.max_objects = int(params['max_objects'])
		self.image_size = int(params['image_size'])
		self.checkpoint_dir = 'checkpoints/'
		self.summary_dir = 'summary/'
		self.max_iterations = int(params['max_iterations'])
		self.investigate = params['investigate']

		self.net = net
		self.dataset_builder = dataset_builder
		#construct graph
		self.construct_graph()

	def _train(self):
		""" create an optimizer and apply to all trainable variables.
		Returns:
			train_op: optimizer for training
		"""

		lr = tf.train.exponential_decay(self.learning_rate, self.global_step, 20, self.moment, staircase=True)
		opt = tf.train.GradientDescentOptimizer(lr)
		grads = opt.compute_gradients(self.total_loss)

		tf.summary.scalar('learning_rate', lr)

		apply_gradient_op = opt.apply_gradients(grads, global_step=self.global_step)

		return apply_gradient_op

	def construct_graph(self):
		# construct graph
		self.global_step = self.net.global_step()

		self.images = tf.placeholder(tf.float32, (self.batch_size, self.image_size, self.image_size, 3))
		self.labels = tf.placeholder(tf.float32, (self.batch_size, self.max_objects, 5))
		self.object_counts = tf.placeholder(tf.int32, (self.batch_size))

		self.predictions = self.net.build(self.images)
		self.total_loss = self.net.loss(self.predictions, self.labels, self.object_counts)
		
		tf.summary.scalar('loss', self.total_loss)
		self.train_op = self._train()


	def solve(self, restore):
		saver = tf.train.Saver(self.net.trainable_collection, write_version=tf.train.SaverDef.V2)
		
		init = tf.global_variables_initializer()

		summary_op = tf.summary.merge_all()
		
#		cluster = tf.train.ClusterSpec({'local': ['localhost:2222']})
#		server = tf.train.Server(cluster, job_name='local', task_index=0)

#		print(server.target)

		sess = tf.Session()

		sess.run(init)
		summary_writer = tf.summary.FileWriter(self.summary_dir, sess.graph)

		latest_checkpoint = tf.train.latest_checkpoint(self.checkpoint_dir)

		if latest_checkpoint != None and restore:
			print('Restoring model `%s`' % latest_checkpoint)
			saver.restore(sess, latest_checkpoint)
		else:
			print('No model found that could be restored')

		#print('global step %d' % tf.train.global_step(sess, self.global_step))
		
		for step in range(self.max_iterations):
			start_time = time.time()
			b_images, b_labels, b_object_counts = self.dataset_builder.batch(self.investigate)
		
			_, summary_str, loss_value, gstep = sess.run([self.train_op, summary_op, self.total_loss, self.global_step], feed_dict={self.images: b_images, self.labels: b_labels, self.object_counts: b_object_counts})
			#loss_value, nilboy = sess.run([self.total_loss, self.nilboy], feed_dict={self.images: np_images, self.labels: np_labels, self.objects_num: np_objects_num})

			print(gstep)
			duration = time.time() - start_time

			assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

			num_examples_per_step = self.batch_size
			examples_per_sec = num_examples_per_step / duration
			sec_per_batch = float(duration)

			format_str = ('step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)')
			print(format_str % (step, loss_value, examples_per_sec, sec_per_batch))

			if step % 10 == 0:
				summary_writer.add_summary(summary_str, step)
			
			if step % 2 == 0 and step != 0:
				print('Saving checkpoint')
				saver.save(sess, self.checkpoint_dir + 'model.ckpt', global_step=self.global_step)

		print('Finished training')

		if restore:			
			saver.save(sess, self.model_dir + 'model.ckpt', global_step=self.global_step)

		sys.stdout.flush()
		sess.close()

