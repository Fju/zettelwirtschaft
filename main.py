from dataset import DatasetBuilder
from train import Solver
from cnn import Net

import tensorflow as tf
import numpy as np
import cv2

import argparse
import time

LABEL_PATH = 'labels'
DATA_DIR = 'train_data'
MODEL_DIR = 'model'

# setup CLI argument parser
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--investigate",
	help="Let user investigate newly created datasets before they are used for training",
	action="store_true")

parser.add_argument("-b", '--batch_size',
	help="Set size of batch",
	default=16,
    type=int)


def process_predicts(predicts):
	class_probs = predicts[0, :, :, :3]
	confidence = predicts[0, :, :, 3:4]
	coordinates = predicts[0, :, :, 4:]

	class_probs = np.reshape(class_probs, (7, 7, 1, 3))
	confidence = np.reshape(confidence, (7, 7, 1, 1))

	P = confidence * class_probs

	coordinates = np.reshape(coordinates, (7, 7, 1, 4))

	indexes = []
	for y in range(7):
		for x in range(7):
			boxes = P[y,x] # shape: (1, 3)
			
			
			index = np.argmax(boxes)
			index = np.unravel_index(index, boxes.shape)
			
			confidence = boxes[index[0],index[1]]

			print(confidence)
			if confidence > 0.01:
				print('found likely box')
				index = np.concatenate([[y, x], index])
				indexes.append(index)
	
	boxes = []
	for i in indexes:
		class_num = i[3]
		max_coordinate = coordinates[i[0], i[1], i[2], :]
		print(max_coordinate)
		cx = max_coordinate[0]
		cy = max_coordinate[1]
		w = max_coordinate[2]
		h = max_coordinate[3]

		cx = (i[1] + cx) * (448/7)
		cy = (i[0] + cy) * (448/7)

		w = w * 448
		h = h * 448

		left = cx - w / 2.0
		top = cy - h / 2.0
		right = left + w
		bottom = top + h
		
		boxes.append([left, top, right, bottom, class_num])	
	
	return boxes

def main():
	args = parser.parse_args()

	print('Parsed arguments', args.investigate, args.batch_size)
	
	# TODO: embed arguments into param dictionary
	
	shared_params = {
		'image_size': 448,
		'num_classes': 3,
		'cell_count': 7,
		'boxes_per_cell': 1,
		'max_objects': 3,
		'batch_size': 16,
		'epochs': 20,
		'max_iterations': 1000,
		'moment': 0.9,
		'learning_rate': 0.01,
		'investigate': False
	}
	
	net = Net(shared_params)
	builder = DatasetBuilder(LABEL_PATH, DATA_DIR, shared_params)
	
	solver = Solver(net, builder, shared_params)

	solver.solve(True)


	# show example
	sess = tf.Session()

	
	
	image = tf.placeholder(tf.float32, (1, 448, 448, 3))
	predicts = net.build(image)

	np_img = cv2.imread('test_image.jpg')
	resized_img = cv2.resize(np_img, (448, 448))
	np_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)

	np_img = np_img.astype(np.float32)
	np_img = 1.0 - (np_img / 255.0)
	np_img = np.reshape(np_img, (1, 448, 448, 3))

	
	saver = tf.train.Saver(net.trainable_collection)

	saver.restore(sess, 'model/model.ckpt-100')
	
	start = time.time()
	np_predict = sess.run(predicts, feed_dict={image: np_img})

	duration = time.time() - start
	print('Time: %.5f' % duration)
	boxes = process_predicts(np_predict)

	for box in boxes:
		class_name = str(box[4])
		print(box)
		cv2.rectangle(resized_img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255))
		cv2.putText(resized_img, class_name, (int(box[0]), int(box[1])), 2, 1.5, (0, 0, 255))
		
	cv2.imshow('result', resized_img)
	cv2.waitKey(0)

if __name__ == '__main__':
		main()
