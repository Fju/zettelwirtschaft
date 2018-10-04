from tensorflow.keras.callbacks import TensorBoard
import os

def find_checkpoint( checkpoint_dir, model_name):
	largest = 0
	checkpoint_path = ''
	for path in os.listdir(checkpoint_dir + '/'):
		if not os.path.isfile(os.path.join(checkpoint_dir, path)):
			# no file, skip
			continue
		
		attr = path.split('.')
		
		if model_name and not(attr[1] == model_name):
			# checkpoint filename doesn't contain the model's name
			continue

		# get step
		n = int(attr[2])
		if n > largest:
			# store largest
			largest = n
			checkpoint_path = os.path.join(checkpoint_dir, path)

	return checkpoint_path



class TensorBoardWrapper(TensorBoard):
	""" dummy that sets the self.validation_data property in order to use the TensorBoard callback with generators. """
	def __init__(self, batch_gen, **kwargs):
		super(TensorBoardWrapper, self).__init__(**kwargs)
		self.batch_gen = batch_gen

	def on_epoch_end(self, epoch, logs):
		# set validation data since tensorboard needs it to write histograms
		x, y = self.batch_gen.get_sample(0)
		self.validation_data = [x, y, np.ones(x.shape[0]), 0.0]
		return super(TensorBoardWrapper, self).on_epoch_end(epoch, logs)

