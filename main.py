from dataset import DatasetBuilder
from train import Solver
from net import Net

import tensorflow as tf
import numpy as np
import cv2

import argparse
import time

#from tensorflow.python.client import device_lib

LABEL_PATH = 'labels'
DATA_DIR = 'train_data'
MODEL_DIR = 'model'

# setup CLI argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--investigate',
	help='Let user investigate newly created datasets before they are used for training',
	action='store_true')

parser.add_argument('-b', '--batch-size',
	help='Set size of batch',
	default=16,
    type=int)

parser.add_argument('-s', '--summarize',
	help='Write summary for tensorboard',
	default=False,
	action='store_true')

parser.add_argument('-r', '--restore',
	help='Decide whether to restore checkpoints to keep on training or start with an "empty" network',
	default=False,
	action='store_true')

parser.add_argument('-t', '--no-training',
	help='Don\'t start training process, just restore latest checkpoint for testing/validation purpose',
	default=False,
	action='store_true')

#device_lib.list_local_devices()

def main():
	args = parser.parse_args()

	print(args)
	
	# TODO: embed arguments into param dictionary
	
	shared_params = {
		'image_size': 448,
		'num_classes': 3,
		'cell_count': 7,
		'boxes_per_cell': 1,
		'max_objects': 3,
		'batch_size': args.batch_size,
		'epochs': 20,
		'max_iterations': 10000,
		'learning_rate_decay': 0.9,
		'learning_rate': 0.01,
		'investigate': args.investigate,
		'summarize': args.summarize
	}
	
	net = Net(shared_params)
	builder = DatasetBuilder(LABEL_PATH, DATA_DIR, shared_params)
	solver = Solver(net, builder, shared_params)

	if not(args.no_training):
		solver.solve(args.restore)
	else:
		# TODO: better method names!
		image, predictions = solver.validate('test_image.jpg')
		builder.validate(image, predictions, threshold=0)


if __name__ == '__main__':
	main()
