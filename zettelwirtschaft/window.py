import cv2
import numpy as np

class WindowHost():
	def __init__(self, params, webcamID=0, width=640, height=480):
		self.cap = cv2.VideoCapture(webcamID)

		self.width = int(self.cap.get(3))
		self.height = int(self.cap.get(4))

		self.image_size = params['image_size']

		if self.width != width or self.height != height:
			self.cap.set(3, width)
			self.cap.set(4, height)

		
	def show(self, model_cb):
		assert self.cap.isOpened()

		while True:
			# receive current frame
			ret, frame = self.cap.read()

			box_top = (self.height - self.image_size) / 2
			box_bottom = (self.height + self.image_size) / 2
			box_left = (self.width - self.image_size) / 2
			box_right = (self.width + self.image_size) / 2
		
			cv2.rectangle(frame, (box_left, box_top), (box_right, box_bottom), (255, 0, 0))
			cv2.imshow('webcam', frame)

			# crop image
			video_in = frame[box_top:box_bottom, box_left:box_right]
			cv2.imshow('video input', video_in)

				
			prediction = model_cb(video_in)		
			cv2.imshow('prediction', prediction)	

			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

		self.cap.release()
		cv2.destroyAllWindows()

