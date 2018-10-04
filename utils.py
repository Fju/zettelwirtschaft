from tensorflow.keras.callbacks import TensorBoard
import os
import json

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


def load_config(config_path):
	# default config file
	dummy = {
		'checkpoint_dir': 'checkpoints',
		'train_data_dir': 'train_data',
		'log_dir': 'logs',
		'learning_rate': 0.001,
		'learning_rate_decay': 0.05,
		'batch_size': 5,
		'num_samples': 10,
		'epochs': 10,
		'verbosity': 1,
		'image_size': 256	
	}

	# check if file exists
	if not(os.path.isfile(config_path)):
		print('Warning: config file is missing, generating default configuration file')
		#write default config file
		with open('./config.json', 'w') as f:
		    json.dump(dummy, f, indent=4)

		return dummy
	else:
		with open(config_path, 'r') as f:
			config_dict = json.load(f)

		complete = True
		for key in dummy:
			# check if config dictionary contains every attribute
			if not(key in config_dict):
				complete = False
				config_dict[key] = dummy[key]

		if not(complete):
			print('Warning: config file is incomplete, start auto-completion')
			with open(config_path, 'w') as f:
				json.dump(config_dict, f, indent=4)
		
		return config_dict

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

