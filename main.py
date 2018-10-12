import argparse
import json
import cv2


from zettelwirtschaft.net import Model
from zettelwirtschaft.dataset import DataGenerator
from zettelwirtschaft.utils import load_config
from zettelwirtschaft.window import WindowHost

from tensorflow.python.client import device_lib


parser = argparse.ArgumentParser()
parser.add_argument('--model-name', default='default', help='model\'s name used for saving (or loading) checkpoints')
parser.add_argument('--config', default='config.json', help='path to configuration file')
parser.add_argument('--list-devices', action='store_true', help='list available devices (GPUs, CPUs, etc.)')
parser.add_argument('--continue', action='store_true', help='loads checkpoint of a model and continues training from that checkpoint')

args = parser.parse_args()
print(args)



def main():
	if args.list_devices:
		print(device_lib.list_local_devices())
		exit()

	params = load_config(args.config)
	
	#training_generator = DataGenerator(params)

	model = Model(args.model_name, params)

	model.load_checkpoint()

	wh = WindowHost(params)

	wh.show(model.evaluate)
	
	#model.train(training_generator)
	


if __name__ == '__main__':
	main()
